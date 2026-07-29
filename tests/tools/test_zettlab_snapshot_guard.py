"""Unit tests for the Zettlab file-change protection pre-mutation hook."""

from __future__ import annotations

import io
import json
import os
import types
import urllib.error

import pytest

from tools import zettlab_snapshot_guard as guard


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Recorder:
    """Stands in for urlopen; records requests and replays scripted replies."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(
            {
                "url": req.full_url,
                "headers": {k.lower(): v for k, v in req.headers.items()},
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
                "timeout": timeout,
            }
        )
        reply = self._replies.pop(0) if self._replies else {"ready": True, "operations": []}
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(json.dumps({"code": 200, "data": reply}).encode("utf-8"))


@pytest.fixture(autouse=True)
def _device_env(monkeypatch, tmp_path):
    """Pretend we are running as a hermes child of local-server on a device."""
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:19090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok123")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    guard.reset_for_test()
    yield
    guard.reset_for_test()


def _install(monkeypatch, *replies):
    rec = _Recorder(replies)
    # 实现走禁用重定向的 _OPENER.open（不是裸 urlopen），mock 也挂在这一层。
    monkeypatch.setattr(guard, "_OPENER", types.SimpleNamespace(open=rec))
    return rec


def test_unguarded_tools_are_ignored(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")
    for name in ("read_file", "search_files", "todo"):
        assert guard.maybe_require_snapshot(name, {"path": str(target)}, turn_id="t1") is None
    assert rec.requests == []


def test_no_op_outside_device_environment(monkeypatch, tmp_path):
    """CLI / dev machines have no local-server callback URL — never block there."""
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="t1") is None
    assert rec.requests == []


def test_write_file_ensures_snapshot_before_mutating(monkeypatch, tmp_path):
    rec = _install(monkeypatch, {"ready": True, "operations": [{"operationId": "aop_1"}]})
    target = tmp_path / "预算.xlsx"
    target.write_text("old")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1") is None

    assert len(rec.requests) == 1
    req = rec.requests[0]
    assert req["url"].endswith("/api/v1/internal/snapshot/agent-protection/ensure")
    assert req["headers"]["x-zettlab-agent-action-token"] == "tok123"
    assert req["body"]["turnId"] == "turn_1"
    assert req["body"]["paths"] == [str(target)]
    assert req["body"]["title"] == "预算.xlsx"
    # The agent identity is never sent — local-server derives it from the token.
    assert "agentId" not in req["body"]


def test_relative_paths_are_resolved_against_cwd(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    (tmp_path / "notes.md").write_text("x")

    guard.maybe_require_snapshot("write_file", {"path": "notes.md"}, turn_id="turn_1")

    assert rec.requests[0]["body"]["paths"] == [str(tmp_path / "notes.md")]


def test_blocks_when_snapshot_is_not_ready(monkeypatch, tmp_path):
    """ready=false 只可能是服务端真的没拍出快照——无保护路径它自己就放行了。"""
    _install(monkeypatch, {"ready": False, "operations": []})
    target = tmp_path / "a.txt"
    target.write_text("x")

    blocked = guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    assert blocked is not None
    assert "NOT modified" in json.loads(blocked)["error"]


def test_unprotected_paths_are_allowed_and_logged(monkeypatch, tmp_path, caplog):
    """无保护放行：不打断用户，但必须在日志里留下「这次没有恢复点」的现场。"""
    _install(monkeypatch, {
        "ready": True,
        "operations": [
            {
                "operationId": "aop_1",
                "state": "unprotected",
                "unprotected": True,
                "unprotectedReason": "storage_low",
            }
        ],
    })
    secret_name = "季度预算.xlsx"
    target = tmp_path / secret_name
    target.write_text("x")

    with caplog.at_level("INFO", logger=guard.logger.name):
        allowed = guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")

    assert allowed is None, "无保护场景直接放行，不阻塞写入"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "no recovery point" in logged
    assert "storage_low" in logged
    assert secret_name not in logged, "日志不记路径，路径只进服务端审计表"


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("http://x", 500, "boom", {}, io.BytesIO(b"{}")),
        # 重定向被 _NoRedirectHandler 拒绝后以 HTTPError 浮出——带 token 的请求
        # 绝不跟去 3xx 指向的地址，fail-closed（Codex review P1）。
        urllib.error.HTTPError("http://x", 302, "moved", {}, io.BytesIO(b"")),
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
    ],
)
def test_fails_closed_on_transport_and_server_errors(monkeypatch, tmp_path, failure):
    _install(monkeypatch, failure)
    target = tmp_path / "a.txt"
    target.write_text("x")

    blocked = guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    assert blocked is not None
    assert "NOT modified" in json.loads(blocked)["error"]


def test_old_local_server_without_endpoint_degrades_open(monkeypatch, tmp_path, caplog):
    """404 means the device firmware predates this feature: proceed, don't lie."""
    _install(monkeypatch, urllib.error.HTTPError("http://x", 404, "nope", {}, io.BytesIO(b"")))
    target = tmp_path / "a.txt"
    target.write_text("x")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1") is None


def test_missing_turn_id_is_fail_closed(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    blocked = guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="")
    assert blocked is not None
    assert rec.requests == []


def test_same_turn_created_files_are_exempt(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    fresh = tmp_path / "generated.md"  # does not exist yet

    # First write creates it: still goes through the entry point, but the file's
    # original state is "absent" so the server takes no snapshot.
    assert guard.maybe_require_snapshot("write_file", {"path": str(fresh)}, turn_id="turn_1") is None
    assert len(rec.requests) == 1

    fresh.write_text("v1")  # the tool actually ran

    # Iterating on its own output must not re-enter the protection path.
    for _ in range(3):
        assert guard.maybe_require_snapshot("write_file", {"path": str(fresh)}, turn_id="turn_1") is None
    assert len(rec.requests) == 1

    # A new turn does not inherit the exemption.
    assert guard.maybe_require_snapshot("write_file", {"path": str(fresh)}, turn_id="turn_2") is None
    assert len(rec.requests) == 2


def test_created_tracking_is_bounded(monkeypatch, tmp_path):
    _install(monkeypatch)
    limit = guard._MAX_CREATED_TRACKED
    for i in range(limit + 5):
        guard.maybe_require_snapshot(
            "write_file", {"path": str(tmp_path / f"f{i}.txt")}, turn_id="turn_1"
        )
    state = guard._states.get("turn_1")
    assert state is not None
    assert len(state.created) <= limit
    assert state.created_overflowed is True


def test_terminal_only_guards_destructive_commands(monkeypatch, tmp_path):
    rec = _install(monkeypatch)

    assert guard.maybe_require_snapshot("terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert rec.requests == []

    assert guard.maybe_require_snapshot("terminal", {"command": "rm -f old.txt"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1
    # Shell paths cannot be parsed reliably; the working directory is reported so
    # the server can snapshot the whole protected folder.
    assert rec.requests[0]["body"]["paths"] == [str(tmp_path)]


def test_v4a_patch_reports_every_touched_path(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x")
    b.write_text("y")
    patch_body = (
        "*** Begin Patch\n"
        f"*** Update File: {a}\n"
        f"*** Delete File: {b}\n"
        "*** End Patch\n"
    )

    guard.maybe_require_snapshot("patch", {"mode": "patch", "patch": patch_body}, turn_id="turn_1")

    assert sorted(rec.requests[0]["body"]["paths"]) == sorted([str(a), str(b)])


def test_patch_replace_reports_its_path(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    target = tmp_path / "a.py"
    target.write_text("x")

    guard.maybe_require_snapshot(
        "patch", {"mode": "replace", "path": str(target), "old_string": "x", "new_string": "y"},
        turn_id="turn_1",
    )
    assert rec.requests[0]["body"]["paths"] == [str(target)]


def test_finish_turn_reports_once_and_only_after_a_snapshot(monkeypatch, tmp_path):
    rec = _install(monkeypatch)

    # Nothing happened this turn → no report at all.
    guard.finish_turn()
    assert rec.requests == []

    target = tmp_path / "a.txt"
    target.write_text("x")
    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    assert len(rec.requests) == 1

    guard.finish_turn("completed")
    assert len(rec.requests) == 2
    finish = rec.requests[1]
    assert finish["url"].endswith("/agent-protection/finish")
    assert finish["body"] == {
        "turnId": "turn_1",
        "state": "completed",
        "errorCode": "",
        "errorStage": "",
    }

    # Repeat finish is a no-op — the turn state was consumed.
    guard.finish_turn("completed")
    assert len(rec.requests) == 2


def test_finish_turn_failure_does_not_raise(monkeypatch, tmp_path):
    """A failed report must never take down the turn — the pin has a TTL."""
    target = tmp_path / "a.txt"
    target.write_text("x")
    _install(monkeypatch, {"ready": True, "operations": []}, urllib.error.URLError("down"))

    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    guard.finish_turn("completed")  # must not raise


def test_action_token_is_never_sent_off_loopback(monkeypatch, tmp_path):
    """The endpoint origin is derived from the injected callback URL only."""
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:19090/api/v1/internal/chat/append")
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    assert rec.requests[0]["url"].startswith("http://127.0.0.1:19090/")


def test_malformed_callback_url_disables_the_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "not-a-url")
    monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1") is None
    assert rec.requests == []


def test_multi_file_title_carries_count(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x")
    b.write_text("y")
    patch_body = f"*** Begin Patch\n*** Update File: {a}\n*** Update File: {b}\n*** End Patch\n"

    guard.maybe_require_snapshot("patch", {"mode": "patch", "patch": patch_body}, turn_id="turn_1")
    assert rec.requests[0]["body"]["title"].endswith("等 2 个文件")


def test_block_logging_records_outcome_without_leaking_paths(monkeypatch, tmp_path, caplog):
    """板子上要能捞到「为什么挡了」，但日志里不能出现用户的文件路径。

    路径只进 local-server 受权限控制的审计表；journalctl 是运维面，任何人读得到。
    """
    _install(monkeypatch, urllib.error.URLError("down"))
    secret_name = "非常机密的季度预算.xlsx"
    target = tmp_path / secret_name
    target.write_text("x")

    with caplog.at_level("WARNING", logger=guard.logger.name):
        blocked = guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")

    assert blocked is not None
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "outcome=" in logged, "运维需要按 outcome 聚合失败原因"
    assert "tool=write_file" in logged
    assert "duration_ms=" in logged
    assert secret_name not in logged
    assert str(tmp_path) not in logged


def test_missing_turn_id_passes_through_outside_device_env(monkeypatch, tmp_path):
    """非设备环境的嵌套 dispatch / MCP bridge 不带 turn_id，也绝不能被挡。"""
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="") is None
    assert rec.requests == []


def test_nested_dispatch_inherits_turn_via_task(monkeypatch, tmp_path):
    """execute_code 沙箱 RPC 二次进入时只带 task_id：凭映射回落到外层轮。"""
    rec = _install(monkeypatch)
    nested = tmp_path / "nested.txt"
    nested.write_text("y")

    # 外层 execute_code dispatch 登记 task → turn（它本身是否触发 ensure 取决
    # 于 execution mode，这里不作假设）。
    assert guard.maybe_require_snapshot(
        "execute_code", {"code": "print(1)"}, turn_id="turn_1", task_id="task_9"
    ) is None

    # 沙箱里 hermes_tools.write_file 回流：turn_id 为空、task_id 相同。
    assert guard.maybe_require_snapshot(
        "write_file", {"path": str(nested)}, turn_id="", task_id="task_9"
    ) is None
    assert rec.requests, "嵌套写入必须触发 ensure"
    assert all(r["body"]["turnId"] == "turn_1" for r in rec.requests)


def test_non_loopback_callback_url_never_receives_token(monkeypatch, tmp_path):
    """回调地址被注入成外部主机时，token 与路径一个字节都不能发出去。"""
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://evil.example.com:9090/api/append")
    monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1") is None
    assert rec.requests == []


def test_concurrent_turns_are_tracked_and_finished_independently(monkeypatch, tmp_path):
    """并发轮互不覆盖；finish 指名收自己的轮，不动别人的 pin。"""
    rec = _install(monkeypatch)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")

    guard.maybe_require_snapshot("write_file", {"path": str(a)}, turn_id="turn_a")
    guard.maybe_require_snapshot("write_file", {"path": str(b)}, turn_id="turn_b")
    assert len(rec.requests) == 2

    guard.finish_turn("completed", turn_id="turn_a")
    assert len(rec.requests) == 3
    assert rec.requests[2]["body"]["turnId"] == "turn_a"

    # turn_b 的状态不受影响，仍能正常收尾。
    guard.finish_turn("failed", turn_id="turn_b")
    assert len(rec.requests) == 4
    assert rec.requests[3]["body"] == {
        "turnId": "turn_b", "state": "failed", "errorCode": "", "errorStage": "",
    }


def test_ambiguous_finish_leaves_concurrent_turns_to_ttl(monkeypatch, tmp_path):
    """多轮并发时不带 turn_id 的 finish 宁可不收——错收会提前解掉别人的 pin。"""
    rec = _install(monkeypatch)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")
    guard.maybe_require_snapshot("write_file", {"path": str(a)}, turn_id="turn_a")
    guard.maybe_require_snapshot("write_file", {"path": str(b)}, turn_id="turn_b")

    guard.finish_turn("completed")
    assert len(rec.requests) == 2  # 没有 finish 请求发出

    guard.finish_turn("completed", turn_id="turn_a")
    guard.finish_turn("completed", turn_id="turn_b")
    assert len(rec.requests) == 4


def test_single_tracked_turn_finishes_without_explicit_id(monkeypatch, tmp_path):
    """cron 拿不到 agent 实例时兜底：只剩一轮在跟踪就收它。"""
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")
    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")

    guard.finish_turn("completed")
    assert rec.requests[-1]["body"]["turnId"] == "turn_1"


def test_write_paths_resolved_via_task_registry(monkeypatch, tmp_path):
    """相对路径按 task 会话注册的 cwd 解析，与文件工具的实际落点一致。"""
    file_tools = pytest.importorskip("tools.file_tools")
    session_dir = tmp_path / "session-cwd"
    session_dir.mkdir()
    resolved = session_dir / "notes.md"
    resolved.write_text("x")

    def fake_resolver(path, task_id="default"):
        assert task_id == "task_9"
        return session_dir / path

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", fake_resolver)
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "write_file", {"path": "notes.md"}, turn_id="turn_1", task_id="task_9"
    )
    assert rec.requests[0]["body"]["paths"] == [str(resolved)]


def test_dns_spoofed_loopback_host_is_rejected(monkeypatch, tmp_path):
    """127. 开头的 DNS hostname 不是 loopback，token 一个字节都不能发出去。"""
    for url in (
        "http://127.evil.example:9090/api/append",
        "http://127.0.0.1.attacker:9090/api/append",
    ):
        monkeypatch.setenv("ZET_CHAT_APPEND_URL", url)
        monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
        guard.reset_for_test()
        rec = _install(monkeypatch)
        target = tmp_path / "a.txt"
        target.write_text("x")
        assert guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="t1") is None
        assert rec.requests == [], url


def test_finish_with_unknown_turn_does_not_steal_other_turns(monkeypatch, tmp_path):
    """带明确 turn_id 却未命中（该轮没写过文件）时，绝不错收唯一余轮的 pin。"""
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")
    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_writer")
    assert len(rec.requests) == 1

    guard.finish_turn("completed", turn_id="turn_reader_only")
    assert len(rec.requests) == 1  # 没有 finish 请求发出

    guard.finish_turn("completed", turn_id="turn_writer")
    assert len(rec.requests) == 2
    assert rec.requests[1]["body"]["turnId"] == "turn_writer"


def test_terminal_append_redirect_triggers_protection(monkeypatch, tmp_path):
    """>> 追加与 tee 同样修改已有文件，必须触发保护。"""
    rec = _install(monkeypatch)
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "printf 'x' >> notes.md"}, turn_id="turn_1"
    ) is None
    assert len(rec.requests) == 1
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "echo hi | tee -a notes.md"}, turn_id="turn_1"
    ) is None
    assert len(rec.requests) == 2


def test_v4a_no_space_header_and_move_are_extracted(monkeypatch, tmp_path):
    """***Update File:（无空格）与 Move File 的两个端点都要被抽出来。"""
    rec = _install(monkeypatch)
    a = tmp_path / "a.py"
    a.write_text("x")
    src = tmp_path / "old.py"
    src.write_text("y")
    dst = tmp_path / "new.py"
    patch_body = (
        "*** Begin Patch\n"
        f"***Update File: {a}\n"
        f"*** Move File: {src} -> {dst}\n"
        "*** End Patch\n"
    )
    guard.maybe_require_snapshot("patch", {"mode": "patch", "patch": patch_body}, turn_id="turn_1")
    assert sorted(rec.requests[0]["body"]["paths"]) == sorted([str(a), str(src), str(dst)])


def test_terminal_workdir_prefers_session_cwd(monkeypatch, tmp_path):
    """destructive terminal 按会话自己的 cwd 记录上报，而不是进程 env。"""
    terminal_tool = pytest.importorskip("tools.terminal_tool")
    session_dir = tmp_path / "session-cwd"
    session_dir.mkdir()
    monkeypatch.setattr(
        terminal_tool, "get_session_cwd",
        lambda key: str(session_dir) if key == "task_9" else None,
    )
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f old.txt"}, turn_id="turn_1", task_id="task_9"
    )
    assert rec.requests[0]["body"]["paths"] == [str(session_dir)]


def test_project_execute_code_protects_session_cwd(monkeypatch, tmp_path):
    """project 模式 execute_code 启动前保护实际 session cwd——脚本能用 Python
    直接改用户文件而不经过任何文件工具。"""
    code_tool = pytest.importorskip("tools.code_execution_tool")
    session_dir = tmp_path / "proj"
    session_dir.mkdir()
    monkeypatch.setattr(code_tool, "_get_execution_mode", lambda: "project")
    monkeypatch.setattr(
        code_tool, "_resolve_child_cwd",
        lambda mode, staging, task_id="": str(session_dir),
    )
    rec = _install(monkeypatch)

    assert guard.maybe_require_snapshot(
        "execute_code", {"code": "open('x','w')"}, turn_id="turn_1", task_id="task_9"
    ) is None
    assert rec.requests[0]["body"]["paths"] == [str(session_dir)]


def test_strict_execute_code_is_not_guarded(monkeypatch, tmp_path):
    """strict 模式的脚本只能经沙箱 RPC 写文件（那条路已被 write_file/patch 覆盖）。"""
    code_tool = pytest.importorskip("tools.code_execution_tool")
    monkeypatch.setattr(code_tool, "_get_execution_mode", lambda: "strict")
    rec = _install(monkeypatch)
    assert guard.maybe_require_snapshot(
        "execute_code", {"code": "print(1)"}, turn_id="turn_1", task_id="task_9"
    ) is None
    assert rec.requests == []


def test_env_is_read_through_secret_scope(monkeypatch, tmp_path):
    """multiplex 下必须走 profile scope，否则会读到别的 agent 的 token。"""
    seen = {}

    def fake_get_secret(name, default=None):
        seen[name] = True
        return os.environ.get(name, default)

    monkeypatch.setattr("agent.secret_scope.get_secret", fake_get_secret)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    assert "ZETTLAB_AGENT_ACTION_TOKEN" in seen
    assert len(rec.requests) == 1


def test_terminal_defaults_to_protection_when_not_provably_readonly(monkeypatch, tmp_path):
    """黑名单列不全会写文件的命令（python -c / git apply / tar / unzip…）：
    证明不了只读就按 cwd 保护。"""
    rec = _install(monkeypatch)
    for cmd in (
        "python -c \"open('x','w').write('1')\"",
        "git apply change.patch",
        "tar -xf archive.tar",
        "unzip -o bundle.zip",
    ):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 4


def test_provably_readonly_commands_skip_protection(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    for cmd in ("ls -la", "cat a.txt | grep foo", "git status", "find . -name x"):
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert rec.requests == []


def test_background_destructive_terminal_is_blocked(monkeypatch, tmp_path):
    """后台破坏性命令会跑到 turn 结束、pin 释放之后：保护窗口对不上就不放行。"""
    rec = _install(monkeypatch)
    blocked = guard.maybe_require_snapshot(
        "terminal", {"command": "rm -rf data", "background": True}, turn_id="turn_1"
    )
    assert blocked is not None
    assert "foreground" in json.loads(blocked)["error"]
    assert rec.requests == []

    # 只读后台命令不受影响。
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la", "background": True}, turn_id="turn_1"
    ) is None
    assert rec.requests == []


def test_absolute_targets_outside_cwd_get_ancillary_protection(monkeypatch, tmp_path):
    """命令里的绝对路径目标（rm /home/...）也要尽力保护，不只 cwd。"""
    rec = _install(monkeypatch)
    outside = tmp_path / "outside-cwd" / "photo.jpg"
    outside.parent.mkdir()
    outside.write_text("x")

    guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -f {outside}"}, turn_id="turn_1"
    )
    assert len(rec.requests) == 2
    assert rec.requests[0]["body"]["paths"] == [str(tmp_path)]
    assert rec.requests[1]["body"]["paths"] == [str(outside)]


def test_ancillary_ensure_failure_does_not_block(monkeypatch, tmp_path):
    """附加保护是加餐：范围外 403 / 服务错误只跳过，不影响已就绪的 cwd 保护。"""
    outside = tmp_path / "other" / "f.txt"
    outside.parent.mkdir()
    outside.write_text("x")
    rec = _install(
        monkeypatch,
        {"ready": True, "operations": []},
        urllib.error.HTTPError("http://x", 403, "scope", {}, io.BytesIO(b"{}")),
    )

    assert guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -f {outside}"}, turn_id="turn_1"
    ) is None
    assert len(rec.requests) == 2


def test_nonexistent_workdir_falls_back_to_host_cwd(monkeypatch, tmp_path):
    """Docker 容器路径（/workspace/...）在 host 上不存在：退回 host env cwd，
    不能把快照拍到不存在的目录上。"""
    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f x", "workdir": "/workspace/proj"}, turn_id="turn_1"
    )
    assert rec.requests[0]["body"]["paths"] == [str(tmp_path)]
