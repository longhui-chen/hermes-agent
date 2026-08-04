"""Unit tests for the Zettlab file-change protection pre-mutation hook."""

from __future__ import annotations

import io
import json
import os
import sys
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
    # HOME 隔离到 tmp：开发机真实 HOME 里的 .bashrc 会触发 rc 遮蔽检测，把
    # 只读放行整体关掉，污染与之无关的用例。
    monkeypatch.setenv("HOME", str(tmp_path))
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


def test_trusted_video_plan_migrate_does_not_snapshot_gateway_cwd(
    monkeypatch, tmp_path
):
    from tools import terminal_tool

    rec = _install(monkeypatch)
    monkeypatch.setattr(
        terminal_tool,
        "_parse_video_edit_runtime_command",
        lambda _command: types.SimpleNamespace(
            argv=[
                sys.executable,
                "/trusted/preference_resolver.py",
                "plan-migrate",
                "--scene",
                "general",
            ]
        ),
    )

    command = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/'
        'scripts/preference_resolver.py" plan-migrate --scene general'
    )
    assert guard.maybe_require_snapshot(
        "terminal", {"command": command}, turn_id="turn_1"
    ) is None
    assert rec.requests == []


def test_trusted_video_helper_snapshots_explicit_write_paths_not_gateway_cwd(
    monkeypatch, tmp_path
):
    from tools import terminal_tool

    rec = _install(monkeypatch, {"ready": True, "operations": []})
    state = tmp_path / "agent" / "workflow_state.json"
    output = tmp_path / "agent" / "vewm_1.mp4"
    monkeypatch.setattr(
        terminal_tool,
        "_parse_video_edit_runtime_command",
        lambda _command: types.SimpleNamespace(
            argv=[
                sys.executable,
                "/trusted/normalize.py",
                "--workflow-state",
                str(state),
                "--input",
                "/volume1/subvol/data/source.mov",
                "--output",
                str(output),
            ]
        ),
    )

    command = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/'
        'scripts/normalize.py" --workflow-state "state" --input "source" '
        '--output "output"'
    )
    assert guard.maybe_require_snapshot(
        "terminal", {"command": command}, turn_id="turn_1"
    ) is None
    assert rec.requests[0]["body"]["paths"] == [str(state), str(output)]
    assert str(tmp_path) not in rec.requests[0]["body"]["paths"]


def test_trusted_video_helper_keeps_new_out_of_scope_writes_fail_closed(
    monkeypatch
):
    from tools import terminal_tool

    rec = _install(monkeypatch, _scope_denied_error())
    monkeypatch.setattr(
        terminal_tool,
        "_parse_video_edit_runtime_command",
        lambda _command: types.SimpleNamespace(
            argv=[
                sys.executable,
                "/trusted/normalize.py",
                "--workflow-state",
                "/etc/new-state.json",
                "--output",
                "/etc/new-output.mp4",
            ]
        ),
    )

    blocked = guard.maybe_require_snapshot(
        "terminal",
        {"command": "python3 trusted/normalize.py --output /etc/new-output.mp4"},
        turn_id="turn_1",
    )
    assert blocked is not None
    assert "NOT modified" in json.loads(blocked)["error"]
    assert rec.requests[0]["body"]["paths"] == [
        "/etc/new-state.json",
        "/etc/new-output.mp4",
    ]


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

    guard.finish_turn("completed", turn_id="turn_1")
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
    guard.finish_turn("completed", turn_id="turn_1")
    assert len(rec.requests) == 2


def test_finish_turn_failure_does_not_raise(monkeypatch, tmp_path):
    """A failed report must never take down the turn — the pin has a TTL."""
    target = tmp_path / "a.txt"
    target.write_text("x")
    _install(monkeypatch, {"ready": True, "operations": []}, urllib.error.URLError("down"))

    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    guard.finish_turn("completed", turn_id="turn_1")  # must not raise


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


def test_finish_without_turn_id_never_pops_sole_tracked_turn(monkeypatch, tmp_path):
    """空 id 一律不收：无写入轮（拿不到 agent 实例）收尾时，进程里唯一的状态
    可能属于另一个还在写的轮，「唯一余轮」兜底会提前解掉对方的 pin
    （Codex review P1）。宁可留给服务端 TTL。"""
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")
    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_writer")

    guard.finish_turn("completed")  # 另一个无写入轮的空 id 收尾
    assert len([r for r in rec.requests if r["url"].endswith("/finish")]) == 0

    guard.finish_turn("completed", turn_id="turn_writer")
    assert rec.requests[-1]["body"]["turnId"] == "turn_writer"


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


def test_empty_task_id_resolves_via_default_session(monkeypatch, tmp_path):
    """handler 侧 `write_file_tool(..., task_id="default")`、`get_session_cwd`
    也把空 key 读作 "default"：guard 不能因为 task_id 为空就跳过会话解析退回
    进程级 cwd，否则 default 会话 `cd` 过之后快照落错目录（Codex review P1）。"""
    file_tools = pytest.importorskip("tools.file_tools")
    session_dir = tmp_path / "default-session-cwd"
    session_dir.mkdir()
    resolved = session_dir / "notes.md"
    resolved.write_text("x")

    seen = []

    def fake_resolver(path, task_id="default"):
        seen.append(task_id)
        return session_dir / path

    monkeypatch.setattr(file_tools, "_resolve_path_for_task", fake_resolver)
    rec = _install(monkeypatch)

    # 直连 registry.dispatch 只带 turn_id 的形态：task_id 缺省为空串。
    guard.maybe_require_snapshot("write_file", {"path": "notes.md"}, turn_id="turn_1")

    assert seen == ["default"], "空 task_id 必须按 handler 口径归一成 default"
    assert rec.requests[0]["body"]["paths"] == [str(resolved)]


def test_empty_task_id_terminal_uses_default_session_cwd(monkeypatch, tmp_path):
    """terminal 侧同源：`get_session_cwd` 的 None/空 key 读 "default" 记录，
    guard 跳过调用就会退回进程级 cwd，而命令实际跑在 default 会话 cd 到的目录。"""
    terminal_tool = pytest.importorskip("tools.terminal_tool")
    session_dir = tmp_path / "default-term-cwd"
    session_dir.mkdir()

    seen = []

    def fake_get_session_cwd(session_key):
        seen.append(session_key)
        return str(session_dir)

    monkeypatch.setattr(terminal_tool, "get_session_cwd", fake_get_session_cwd)
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot("terminal", {"command": "rm -f a.txt"}, turn_id="turn_1")

    assert seen and seen[0] in ("", None), "空 task_id 要原样交给 get_session_cwd 归一"
    assert rec.requests[0]["body"]["paths"] == [str(session_dir)]


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
    for cmd in ("ls -la", "cat a.txt | grep foo", "head -n 5 a.txt", "wc -l a.txt"):
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert rec.requests == []


def test_pseudo_readonly_commands_with_write_actions_are_protected(monkeypatch, tmp_path):
    """find -delete / find -exec / sort -o / env <cmd> 都能写文件——这些名字不进
    只读安全清单（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "find . -delete",
        "find . -name '*.tmp' -exec rm {} +",
        "sort -o sorted.txt input.txt",
        "env FOO=1 python do_write.py",
    ):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 4


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX managed-terminal paths")
def test_managed_python_skill_entry_is_not_treated_as_a_write_target(
    monkeypatch, tmp_path
):
    """A managed terminal may read its active profile's skill entrypoint.

    The cwd still gets the normal recovery point, while the read-only Python
    source is omitted from ancillary write targets so an out-of-scope skill
    path cannot reject an otherwise valid command.
    """
    profile_home = tmp_path / "hermes_home" / "profiles" / "agent-a"
    script = profile_home / "skills" / "support-suite" / "scripts" / "onboard.py"
    output = tmp_path / "agents" / "data" / "agent-a" / "output"
    script.parent.mkdir(parents=True)
    output.mkdir(parents=True)
    script.write_text("print('ok')")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    rec = _install(monkeypatch, {"ready": True, "operations": []})

    allowed = guard.maybe_require_snapshot(
        "terminal",
        {"command": f'python3 "{script}"', "workdir": str(output)},
        turn_id="turn_1",
    )

    assert allowed is None
    assert len(rec.requests) == 1
    assert rec.requests[0]["body"]["paths"] == [str(output)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX managed-terminal paths")
def test_managed_python_skill_user_file_argument_remains_protected(
    monkeypatch, tmp_path
):
    """Only the interpreter source is read-only; later path args may be writes."""
    profile_home = tmp_path / "hermes_home" / "profiles" / "agent-a"
    script = profile_home / "skills" / "support-suite" / "scripts" / "onboard.py"
    output = tmp_path / "agents" / "data" / "agent-a" / "output"
    target = tmp_path / "user-data" / "settings.json"
    script.parent.mkdir(parents=True)
    output.mkdir(parents=True)
    target.parent.mkdir()
    script.write_text("print('ok')")
    target.write_text("{}")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    rec = _install(
        monkeypatch,
        {"ready": True, "operations": []},
        {"ready": True, "operations": []},
    )

    allowed = guard.maybe_require_snapshot(
        "terminal",
        {
            "command": f'python3 "{script}" --config "{target}"',
            "workdir": str(output),
        },
        turn_id="turn_1",
    )

    assert allowed is None
    assert len(rec.requests) == 2
    assert rec.requests[1]["body"]["paths"] == [str(target)]


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


def test_opener_disables_proxies_and_redirects(monkeypatch):
    """带 token 的请求既不跟随重定向、也不经代理（Codex review P1）。

    传入 ProxyHandler({}) 会把默认的 env 代理 handler 从 build_opener 里挤掉，
    而空 dict 的 ProxyHandler 自身没有任何协议方法、不会被注册——净效果是
    opener 里**没有任何**代理 handler。
    """
    handlers = list(getattr(guard._OPENER, "handlers", []))
    assert not any(
        isinstance(h, guard.urllib.request.ProxyHandler) for h in handlers
    ), "不该存在任何代理 handler（含按 HTTP(S)_PROXY 环境变量装的默认项）"
    assert any(isinstance(h, guard._NoRedirectHandler) for h in handlers)


def test_container_workspace_paths_map_back_to_host(monkeypatch, tmp_path):
    """Docker 挂载会话下 file_tools 解析出的 /workspace/... 要反解回 host 路径。
    判定按 terminal 后端配置（容器后端 + 挂载开关），host 上是否恰好存在
    /workspace 不影响结果（Codex review P1）。"""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    rec = _install(monkeypatch)
    (tmp_path / "notes.md").write_text("x")

    guard.maybe_require_snapshot(
        "write_file", {"path": "/workspace/notes.md"}, turn_id="turn_1"
    )
    assert rec.requests[0]["body"]["paths"] == [str(tmp_path / "notes.md")]


def test_local_backend_keeps_workspace_path_literal(monkeypatch, tmp_path):
    """local 后端（或挂载开关未开）下 /workspace 就是字面 host 路径，不反解
    ——用 host 存在性做判定会在 host 恰好有 /workspace 时把挂载会话的写入
    错报、在 local 会话里把真实 /workspace 误改写（Codex review P1）。"""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", raising=False)
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "write_file", {"path": "/workspace/notes.md"}, turn_id="turn_1"
    )
    assert rec.requests[0]["body"]["paths"] == ["/workspace/notes.md"]


def test_relative_workdir_resolves_against_session_cwd(monkeypatch, tmp_path):
    """相对 workdir 是「在会话 cwd 下 cd」的语义，必须锚到会话 cwd 而不是进程 env。"""
    terminal_tool = pytest.importorskip("tools.terminal_tool")
    session_dir = tmp_path / "session-cwd"
    sub = session_dir / "subdir"
    sub.mkdir(parents=True)
    # 进程 env 下也有同名目录——错误锚点会解析到这里。
    (tmp_path / "subdir").mkdir()
    monkeypatch.setattr(
        terminal_tool, "get_session_cwd",
        lambda key: str(session_dir) if key == "task_9" else None,
    )
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f x", "workdir": "subdir"},
        turn_id="turn_1", task_id="task_9",
    )
    assert rec.requests[0]["body"]["paths"] == [str(sub)]


def test_strict_execute_code_gets_ancillary_absolute_path_protection(monkeypatch, tmp_path):
    """strict 只换 cwd、无文件系统隔离：脚本里已存在的绝对路径目标要尽力保护，
    且 finish 能释放这些 operation 的 pin。"""
    code_tool = pytest.importorskip("tools.code_execution_tool")
    monkeypatch.setattr(code_tool, "_get_execution_mode", lambda: "strict")
    target = tmp_path / "Documents" / "a.txt"
    target.parent.mkdir()
    target.write_text("x")
    rec = _install(monkeypatch)

    assert guard.maybe_require_snapshot(
        "execute_code", {"code": f"open('{target}','w').write('y')"},
        turn_id="turn_1", task_id="task_9",
    ) is None
    assert len(rec.requests) == 1
    assert rec.requests[0]["body"]["paths"] == [str(target)]

    guard.finish_turn("completed", turn_id="turn_1")
    assert len(rec.requests) == 2
    assert rec.requests[1]["url"].endswith("/agent-protection/finish")


def _scope_denied_error():
    body = json.dumps({
        "error": {"code": "SNAPSHOT_AGENT_PATH_OUT_OF_SCOPE", "message": "path out of scope"},
    }).encode("utf-8")
    return urllib.error.HTTPError(
        "http://127.0.0.1:19090/api/v1/internal/snapshot/agent-protection/ensure",
        403, "Forbidden", None, io.BytesIO(body),
    )


def test_strict_execute_code_blocks_when_ancillary_ensure_fails(monkeypatch, tmp_path):
    """strict 的唯一保护建不成必须 fail-closed：local-server 断连时不放行写入
    （Codex review P1）。"""
    code_tool = pytest.importorskip("tools.code_execution_tool")
    monkeypatch.setattr(code_tool, "_get_execution_mode", lambda: "strict")
    target = tmp_path / "Documents" / "a.txt"
    target.parent.mkdir()
    target.write_text("x")
    _install(monkeypatch, urllib.error.URLError("down"))

    out = guard.maybe_require_snapshot(
        "execute_code", {"code": f"open('{target}','w').write('y')"},
        turn_id="turn_1", task_id="task_9",
    )
    assert out is not None and "NOT executed" in out


def test_strict_execute_code_skips_out_of_scope_but_requires_in_scope(monkeypatch, tmp_path):
    """批量 403（范围外路径混入）时逐路径重试：范围外只跳过（/etc 只读引用不
    误杀脚本），范围内必须建成恢复点（Codex review P1）。"""
    code_tool = pytest.importorskip("tools.code_execution_tool")
    monkeypatch.setattr(code_tool, "_get_execution_mode", lambda: "strict")
    in_scope = tmp_path / "Documents" / "a.txt"
    in_scope.parent.mkdir()
    in_scope.write_text("x")
    out_scope = tmp_path / "etc-hosts"
    out_scope.write_text("127.0.0.1")
    rec = _install(
        monkeypatch,
        _scope_denied_error(),               # 批量：整批 403
        {"ready": True, "operations": []},   # 逐路径：in_scope 建成
        _scope_denied_error(),               # 逐路径：out_scope 范围外跳过
    )

    code = f"open('{in_scope}','w').write('y'); print(open('{out_scope}').read())"
    assert guard.maybe_require_snapshot(
        "execute_code", {"code": code}, turn_id="turn_1", task_id="task_9",
    ) is None
    assert len(rec.requests) == 3
    assert sorted(rec.requests[0]["body"]["paths"]) == sorted([str(in_scope), str(out_scope)])
    assert rec.requests[1]["body"]["paths"] == [str(in_scope)]
    assert rec.requests[2]["body"]["paths"] == [str(out_scope)]

    # in_scope 建成过恢复点：finish 要释放它的 pin。
    guard.finish_turn("completed", turn_id="turn_1")
    assert rec.requests[-1]["url"].endswith("/agent-protection/finish")


def test_quoted_absolute_paths_with_spaces_are_protected(monkeypatch, tmp_path):
    """引号里带空格的绝对路径要完整抽出——裸 token 正则在空格处截断，会漏掉
    真实写入目标（Codex review P1）。"""
    rec = _install(monkeypatch)
    spaced = tmp_path / "My Documents"
    spaced.mkdir()
    doc = spaced / "a.txt"
    doc.write_text("x")

    guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -f '{doc}'"}, turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(doc) in ensured


def test_container_collapsed_task_key_maps_back_to_turn(monkeypatch, tmp_path):
    """Docker/SSH 后端把 dispatch 的 task_id 折叠成容器 key（通常 default）；
    折叠 key 也要登记，否则沙箱 RPC 二次进入查不到轮、被 missing_turn_id 误拒
    （Codex review P1）。"""
    fake_terminal = types.SimpleNamespace(_resolve_container_task_id=lambda t: "default")
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", fake_terminal)
    rec = _install(monkeypatch)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")

    # 外层 dispatch 带原始 task_id + turn，登记原始与折叠两个 key。
    assert guard.maybe_require_snapshot(
        "write_file", {"path": str(a)}, turn_id="turn_1", task_id="task_outer",
    ) is None
    # 沙箱 RPC 二次进入：只带折叠后的容器 key，也必须回落到外层轮。
    assert guard.maybe_require_snapshot("write_file", {"path": str(b)}, task_id="default") is None
    assert rec.requests[-1]["body"]["turnId"] == "turn_1"


def test_all_git_commands_are_protected(monkeypatch, tmp_path):
    """git 整体移出只读清单（Codex review P1 ×N）：它的行为由用户配置驱动，
    diff.external / textconv / core.fsmonitor / core.pager / alias / hooks 都能
    挂上任意外部命令，逐个子命令 flag 去堵是无穷尽的。代价只是 git 命令按 cwd
    拍一张幂等快照（每轮每目录一张），不是阻断。"""
    rec = _install(monkeypatch)
    cmds = (
        "git status",
        "git diff",
        "git diff --no-ext-diff --no-textconv",
        "git log --oneline",
        "git branch",
        "git remote -v",
        "git rev-parse HEAD",
    )
    for cmd in cmds:
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == len(cmds)


def test_shared_collapsed_key_with_concurrent_turns_fails_closed(monkeypatch, tmp_path):
    """共享容器把并发轮折叠到同一个 key：归属不可判定时受保护写入必须
    fail-closed（覆盖式登记会把 ensure 归错轮、随对方 finish 提前解 pin，
    Codex review P1）；一轮结束后剩下的那轮重新可归属。"""
    fake_terminal = types.SimpleNamespace(_resolve_container_task_id=lambda t: "default")
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", fake_terminal)
    rec = _install(monkeypatch)
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for f in (a, b, c):
        f.write_text("x")

    assert guard.maybe_require_snapshot(
        "write_file", {"path": str(a)}, turn_id="turn_1", task_id="task_a",
    ) is None
    assert guard.maybe_require_snapshot(
        "write_file", {"path": str(b)}, turn_id="turn_2", task_id="task_b",
    ) is None

    # 两轮都在册：折叠 key 分不清归属，嵌套写入 fail-closed。
    blocked = guard.maybe_require_snapshot("write_file", {"path": str(c)}, task_id="default")
    assert blocked is not None and "cannot be attributed" in blocked

    # 一轮结束后恢复可归属：写入归到仍在进行的那一轮。
    guard.finish_turn("completed", turn_id="turn_1")
    assert guard.maybe_require_snapshot("write_file", {"path": str(c)}, task_id="default") is None
    assert rec.requests[-1]["body"]["turnId"] == "turn_2"


def test_home_and_parent_relative_write_targets_get_ancillary_protection(monkeypatch, tmp_path):
    """`$HOME/...`、`~/...`、`../...` 都能指到 cwd 之外的用户文件：附加保护要
    按 profile home / 会话工作目录展开后接住（Codex review P1）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    home_doc = home / "Documents" / "a.txt"
    home_doc.write_text("x")
    monkeypatch.setenv("HOME", str(home))

    outside = tmp_path / "outside.txt"
    outside.write_text("y")
    sub = tmp_path / "sub"
    sub.mkdir()

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal",
        {"command": 'rm -f "$HOME/Documents/a.txt" ../outside.txt', "workdir": str(sub)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(home_doc) in ensured, "HOME 前缀目标要展开后保护"
    assert str(outside) in ensured, "../ 相对目标要按工作目录展开后保护"


def test_explicit_workspace_volume_maps_to_volume_host(monkeypatch, tmp_path):
    """docker_volumes 显式挂 /workspace 时优先于 cwd bind（DockerEnvironment
    同序）：/workspace 要反解到该 volume 的 host 侧（Codex review P1）。"""
    vol_host = tmp_path / "vol"
    vol_host.mkdir()
    (vol_host / "notes.md").write_text("x")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{vol_host}:/workspace"]))
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "write_file", {"path": "/workspace/notes.md"}, turn_id="turn_1"
    )
    assert rec.requests[0]["body"]["paths"] == [str(vol_host / "notes.md")]


def test_text_to_speech_custom_output_path_is_protected(monkeypatch, tmp_path):
    """text_to_speech 的自定义 output_path 会先删再写：已存在的用户文件要有
    恢复点；默认输出（无 output_path）不涉用户文件、零请求（Codex review P1）。"""
    rec = _install(monkeypatch)
    target = tmp_path / "Documents"
    target.mkdir()
    doc = target / "a.txt"
    doc.write_text("x")

    assert guard.maybe_require_snapshot(
        "text_to_speech", {"text": "hi", "output_path": str(doc)}, turn_id="turn_1"
    ) is None
    assert rec.requests[0]["body"]["paths"] == [str(doc)]

    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "text_to_speech", {"text": "hi"}, turn_id="turn_1"
    ) is None
    assert len(rec.requests) == 1, "默认输出不该发起保护请求"


def test_self_backgrounding_write_commands_are_blocked(monkeypatch, tmp_path):
    """shell 自行后台化的写入（结尾 &、nohup / setsid）与 background=true 同罪：
    finish 解 pin 时子进程可能仍在写（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "rm -rf data &",
        "nohup sh -c 'rm -f x'",
        "setsid rm -f x",
        "cp a b & cp c d",
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # `&&` / `2>&1` / `&>` 不是后台化；只读命令带 & 也不进这条路。
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f x && rm -f y"}, turn_id="turn_1") is None
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f x 2>&1"}, turn_id="turn_1") is None
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la &"}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "非后台写入命令照常走保护"


def test_docker_volume_paths_map_back_to_host_longest_prefix(monkeypatch, tmp_path):
    """docker_volumes 的每一条 host bind 都要能反解（不止 /workspace）：容器内
    写 /mnt/pics 动的是 host 侧挂载源；嵌套挂载按最长前缀取（Codex review P1）。"""
    pics = tmp_path / "Pictures"
    nested = tmp_path / "Nested"
    pics.mkdir()
    nested.mkdir()
    (pics / "a.jpg").write_text("x")
    (nested / "b.jpg").write_text("y")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([
        f"{pics}:/mnt/pics",
        f"{nested}:/mnt/pics/nested",
    ]))
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot("write_file", {"path": "/mnt/pics/a.jpg"}, turn_id="turn_1")
    assert rec.requests[0]["body"]["paths"] == [str(pics / "a.jpg")]

    guard.maybe_require_snapshot("write_file", {"path": "/mnt/pics/nested/b.jpg"}, turn_id="turn_1")
    assert rec.requests[1]["body"]["paths"] == [str(nested / "b.jpg")], "嵌套挂载要按最长前缀反解"


def test_tts_format_realigned_sibling_is_protected(monkeypatch, tmp_path):
    """command TTS provider 会把 output_path 后缀改写成配置的 output_format 再
    删 / 写：同名的四种合法格式变体一并纳保（Codex review P1）。"""
    rec = _install(monkeypatch)
    voice_mp3 = tmp_path / "voice.mp3"
    voice_wav = tmp_path / "voice.wav"
    voice_mp3.write_text("m")
    voice_wav.write_text("w")

    guard.maybe_require_snapshot(
        "text_to_speech", {"text": "hi", "output_path": str(voice_mp3)}, turn_id="turn_1"
    )
    paths = rec.requests[0]["body"]["paths"]
    assert str(voice_mp3) in paths and str(voice_wav) in paths


def test_env_prefixed_commands_are_not_provably_readonly(monkeypatch, tmp_path):
    """env 赋值能改写命令行为（GIT_EXTERNAL_DIFF=rm git diff 会对 diff 路径执行
    rm）：带 env 前缀的命令不再证明只读，按 cwd 保护（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "GIT_EXTERNAL_DIFF=rm git diff",
        "PAGER=x LESSOPEN='|rm %s' less a.txt",
    ):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "带 env 前缀的命令要走保护"

    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "cat a.txt"}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "无 env 前缀的可证明只读命令仍免保护"


def test_extra_args_bind_mounts_map_back_to_host(monkeypatch, tmp_path):
    """docker_extra_args 里的 -v / --mount type=bind 会被原样追加进 docker run：
    这些挂载也要进反解表（Codex review P1）。"""
    host_v = tmp_path / "Pictures"
    host_m = tmp_path / "Music"
    host_v.mkdir()
    host_m.mkdir()
    (host_v / "a.jpg").write_text("x")
    (host_m / "b.mp3").write_text("y")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", json.dumps([
        "-v", f"{host_v}:/mnt/pics",
        "--mount", f"type=bind,source={host_m},target=/mnt/music,readonly",
    ]))
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot("write_file", {"path": "/mnt/pics/a.jpg"}, turn_id="turn_1")
    assert rec.requests[0]["body"]["paths"] == [str(host_v / "a.jpg")]
    guard.maybe_require_snapshot("write_file", {"path": "/mnt/music/b.mp3"}, turn_id="turn_1")
    assert rec.requests[1]["body"]["paths"] == [str(host_m / "b.mp3")]


def test_ancillary_container_paths_are_mapped_before_filtering(monkeypatch, tmp_path):
    """命令文本里的容器口径绝对路径要先反解再做 lexists 过滤：按 host 字面量
    过滤会把挂载目录下的目标静默漏掉（Codex review P1）。"""
    pics = tmp_path / "Pictures"
    pics.mkdir()
    target = pics / "a.jpg"
    target.write_text("x")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{pics}:/mnt/pics"]))
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f /mnt/pics/a.jpg"}, turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(target) in ensured, "容器路径要反解到 host 后纳入附加保护"


def test_amp_separated_write_segment_is_not_readonly(monkeypatch, tmp_path):
    """单个 & 是 control operator：`ls & rm x` 的写入段不能藏进只读判定
    （Codex review P1）。混有后台段的写入命令按自后台化阻断。"""
    rec = _install(monkeypatch)
    for cmd in ("ls & rm -f old.txt", "true & rm -rf data"):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # 纯只读的管道 / 链不受影响。
    for cmd in ("cat a.txt | grep foo", "ls -la && wc -l a.txt"):
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert rec.requests == []


def test_home_expansion_uses_subprocess_home(monkeypatch, tmp_path):
    """home 目标要按工具子进程实际生效的 HOME 展开（home_mode=profile / 容器
    fallback 时 HOME 被换成 {HERMES_HOME}/home），进程 HOME 会指错目标
    （Codex review P1）。"""
    import hermes_constants

    proc_home = tmp_path / "proc-home"
    sub_home = tmp_path / "profile-home"
    (proc_home / "Documents").mkdir(parents=True)
    (sub_home / "Documents").mkdir(parents=True)
    (proc_home / "Documents" / "a.txt").write_text("proc")
    real_target = sub_home / "Documents" / "a.txt"
    real_target.write_text("sub")
    monkeypatch.setenv("HOME", str(proc_home))
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda env=None: str(sub_home))

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": 'rm -f "$HOME/Documents/a.txt"'},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(real_target) in ensured, "要按子进程 HOME 展开"
    assert str(proc_home / "Documents" / "a.txt") not in ensured


def test_path_qualified_executables_are_not_readonly(monkeypatch, tmp_path):
    """`./ls`、/tmp/cat 执行的是任意程序，与同名系统命令无关：只对无路径分隔符
    的命令名做只读放行（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in ("./ls", "/tmp/cat a.txt", "../bin/grep foo f", "bin/less x"):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 4

    guard.reset_for_test()
    for cmd in ("ls -la", "cat a.txt"):
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 4, "无路径的系统命令仍免保护"


def test_config_driven_commands_are_always_protected(monkeypatch, tmp_path):
    """less/more 与 git 一样整体移出只读清单：LESSOPEN / LESSCLOSE 预处理器、
    diff.external / core.fsmonitor 等既有用户配置都能让「看起来只读」的命令执行
    任意外部程序（Codex review P1 ×N）。"""
    rec = _install(monkeypatch)
    cmds = (
        "less a.txt",
        "printf data | less -O /tmp/notes.txt",
        "more a.txt",
        "git status",
        "git diff",
    )
    for cmd in cmds:
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == len(cmds)


def test_glob_write_targets_are_expanded_before_protection(monkeypatch, tmp_path):
    """glob 目标（rm -rf /home/a/Doc*）要按 shell 语义展开后逐个保护，不展开会
    让真实目标静默漏掉（Codex review P1）。"""
    docs = tmp_path / "Documents"
    docs.mkdir()
    a = docs / "a.txt"
    b = docs / "b.txt"
    a.write_text("x")
    b.write_text("y")
    sub = tmp_path / "cwd"
    sub.mkdir()

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -rf {docs}/*", "workdir": str(sub)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(a) in ensured and str(b) in ensured
    assert str(docs) in ensured, "父目录一并纳保（展开结果被删后目录本身也变了）"


def test_tts_output_path_uses_main_process_semantics(monkeypatch, tmp_path):
    """TTS 在主进程里 Path(output_path).expanduser() 落盘：相对路径锚进程 cwd、
    不做容器反解——guard 必须用同一套语义（Codex review P1）。"""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{tmp_path}:/mnt/audio"]))
    monkeypatch.chdir(tmp_path)
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "text_to_speech", {"text": "hi", "output_path": "voice.mp3"},
        turn_id="turn_1", task_id="task_9",
    )
    assert rec.requests[0]["body"]["paths"] == [str(tmp_path / "voice.mp3")]

    # 容器口径的字面路径也按主进程语义（不反解到 volume host 侧）。
    guard.maybe_require_snapshot(
        "text_to_speech", {"text": "hi", "output_path": "/mnt/audio/x.mp3"},
        turn_id="turn_1", task_id="task_9",
    )
    assert rec.requests[1]["body"]["paths"] == ["/mnt/audio/x.mp3"]


def test_exported_shell_function_shadow_disables_readonly(monkeypatch, tmp_path):
    """导出的 bash function（BASH_FUNC_ls%%=...）会取代同名系统命令，且随 env
    传进 terminal 子进程——命中即不判只读（Codex review P1）。"""
    rec = _install(monkeypatch)
    assert guard.maybe_require_snapshot("terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert rec.requests == []

    guard.reset_for_test()
    monkeypatch.setenv("BASH_FUNC_ls%%", "() { rm -f victim; }")
    assert guard.maybe_require_snapshot("terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "被导出函数遮蔽的命令要走 cwd 保护"


def test_home_and_parent_globs_are_expanded(monkeypatch, tmp_path):
    """`rm -rf ~/Doc*` / `rm -rf ../Doc*` 的裸 token 也要带 glob 字符并展开，
    否则截断成不存在的字面量后被静默丢掉（Codex review P1）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    home_doc = home / "Documents" / "a.txt"
    home_doc.write_text("x")
    monkeypatch.setenv("HOME", str(home))

    sibling = tmp_path / "Docs-sibling"
    sibling.mkdir()
    (sibling / "b.txt").write_text("y")
    cwd = tmp_path / "Work"
    cwd.mkdir()

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -rf ~/Doc* ../Docs-sib*", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(home / "Documents") in ensured, "~ glob 要展开"
    assert str(sibling) in ensured, "../ glob 要展开"


def test_unbound_profile_scope_fails_closed(monkeypatch, tmp_path):
    """multiplex 下 profile scope 未绑定 = 判定不了，不是「不是设备环境」：
    放行会让该 profile 的用户文件无恢复点被改（Codex review P1）。"""
    class _Unscoped(RuntimeError):
        pass
    _Unscoped.__name__ = "UnscopedSecretError"

    fake_scope = types.SimpleNamespace(
        get_secret=lambda name, default=None: (_ for _ in ()).throw(_Unscoped("no scope bound")),
        is_multiplex_active=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "agent.secret_scope", fake_scope)
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    blocked = guard.maybe_require_snapshot(
        "write_file", {"path": str(target)}, turn_id="turn_1")
    assert blocked is not None and "NOT modified" in json.loads(blocked)["error"]
    assert rec.requests == []


def test_terminal_ancillary_failure_blocks_the_command(monkeypatch, tmp_path):
    """terminal 从一个 cwd 写另一个受保护目录时，附加路径建不出恢复点要阻断
    ——只记日志放行等于让写入无恢复点发生（Codex review P1）。"""
    outside = tmp_path / "Documents"
    outside.mkdir()
    doc = outside / "a.txt"
    doc.write_text("x")
    cwd = tmp_path / "Work"
    cwd.mkdir()

    # 主 cwd ensure 成功，附加路径 ensure 传输失败。
    rec = _install(
        monkeypatch,
        {"ready": True, "operations": []},
        urllib.error.URLError("down"),
    )
    blocked = guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -f {doc}", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    assert blocked is not None and "NOT executed" in blocked
    assert len(rec.requests) == 2


def test_glob_expansion_is_streamed_and_capped(monkeypatch, tmp_path):
    """海量匹配的 glob 要流式截断，不能先构造完整列表——端侧 2GB 预算下 guard
    自己会卡住或 OOM（HR1 / Codex review P1）。父目录优先保住。"""
    big = tmp_path / "Pictures"
    big.mkdir()
    for i in range(200):
        (big / f"p{i:03d}.jpg").write_text("x")
    cwd = tmp_path / "Work"
    cwd.mkdir()

    calls = {"n": 0}
    real_iglob = guard.glob.iglob

    def counting_iglob(pattern, *a, **kw):
        for item in real_iglob(pattern, *a, **kw):
            calls["n"] += 1
            yield item

    monkeypatch.setattr(guard.glob, "iglob", counting_iglob)
    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -rf {big}/*", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(big) in ensured, "父目录必须纳保（覆盖面最大）"
    assert len(ensured) <= guard._MAX_ANCILLARY_PATHS + 1
    assert calls["n"] <= guard._MAX_ANCILLARY_PATHS, "不该枚举完整结果集"


def test_rg_is_no_longer_readonly(monkeypatch, tmp_path):
    """rg 的 --pre 可以来自 RIPGREP_CONFIG_PATH 指向的配置文件（只有 --no-config
    忽略它）：与 git / less 同理，配置驱动的命令整体移出只读清单
    （Codex review P1）。grep 族没有等价面，保留。"""
    rec = _install(monkeypatch)
    for cmd in ("rg needle victim.txt", "rg --no-config needle f"):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 2

    guard.reset_for_test()
    for cmd in ("grep -r foo .", "egrep bar f", "fgrep baz f"):
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "grep 族仍然只读"


def test_brace_expansion_targets_are_protected(monkeypatch, tmp_path):
    """`rm -f /home/a/Doc{1,2}.txt` 真会删两个文件：静态可确定的 brace 组要展开
    后再过滤，否则截成不存在的字面量被丢掉（Codex review P1）。"""
    docs = tmp_path / "Documents"
    docs.mkdir()
    one = docs / "Doc1.txt"
    two = docs / "Doc2.txt"
    one.write_text("a")
    two.write_text("b")
    cwd = tmp_path / "Work"
    cwd.mkdir()

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": f"rm -f {docs}/Doc{{1,2}}.txt", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(one) in ensured and str(two) in ensured


def test_quoted_home_concatenation_is_resolved(monkeypatch, tmp_path):
    """shell 把 `"$HOME"/Documents/a.txt` 拼成一个路径词——这是常见的安全写法，
    提取前要归一化，否则整条附加保护抽不到（Codex review P1）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    doc = home / "Documents" / "a.txt"
    doc.write_text("x")
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "Work"
    cwd.mkdir()

    rec = _install(monkeypatch)
    for cmd in (
        'rm -f "$HOME"/Documents/a.txt',
        'rm -f "${HOME}"/Documents/a.txt',
    ):
        guard.reset_for_test()
        rec.requests.clear()
        guard.maybe_require_snapshot(
            "terminal", {"command": cmd, "workdir": str(cwd)},
            turn_id="turn_1", task_id="task_9",
        )
        ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
        assert str(doc) in ensured, cmd


def test_quoted_or_wrapped_daemonizers_are_blocked(monkeypatch, tmp_path):
    """引号 / 包裹层里的 setsid、nohup 在 shell 剥引号后照常执行——朴素空格
    split 会漏判，让写入跑到 finish 解 pin 之后（Codex review P1）。引号不配对
    等解析不了的形态 fail-closed。"""
    rec = _install(monkeypatch)
    for cmd in (
        "'setsid' -f sh -c 'sleep 1; rm -f victim'",
        '"nohup" rm -f victim',
        "env setsid rm -f victim",
        "command setsid rm -f victim",
        "nice -n 10 setsid rm -f victim",
        "rm -f 'victim",  # 引号不配对：识别不准即不放行
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # `command -v setsid` 只查名字不执行；普通写入命令照常走保护。
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "command -v setsid && rm -f x"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "非后台写入命令照常走保护"


def test_adjacent_quoted_path_segments_are_concatenated(monkeypatch, tmp_path):
    """shell 会把相邻 quoted / unquoted 段拼成一个 word：`"$HOME"/"Documents"/a`
    删的是 HOME 下的真实文件，逐段正则在引号处断开就抽不到附加目标
    （Codex review P1）。"""
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    doc_a = home / "Documents" / "a.txt"
    doc_a.write_text("x")
    doc_b = home / "Documents" / "b.txt"
    doc_b.write_text("y")
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "Work"
    cwd.mkdir()

    rec = _install(monkeypatch)
    for cmd, target in (
        ('rm -f "$HOME"/"Documents/a.txt"', doc_a),
        ('rm -f "$HOME"/"Documents"/b.txt', doc_b),
        ("rm -f '%s'/'Documents'/a.txt" % home, doc_a),
    ):
        guard.reset_for_test()
        rec.requests.clear()
        guard.maybe_require_snapshot(
            "terminal", {"command": cmd, "workdir": str(cwd)},
            turn_id="turn_1", task_id="task_9",
        )
        ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
        assert str(target) in ensured, cmd


def test_terminal_workdir_bridges_lazy_terminal_cwd(monkeypatch, tmp_path):
    """local backend 无 session_cwd / 显式 workdir 时兜底读 TERMINAL_CWD——它
    是懒桥接的，不先触发 _get_env_config()，config.yaml 的 terminal.cwd 根本不
    在环境里，guard 会给 Hermes 进程 cwd 建快照而命令实际跑在 terminal.cwd
    （Codex review P1）。"""
    terminal_tool = pytest.importorskip("tools.terminal_tool")
    proj = tmp_path / "proj"
    proj.mkdir()

    def fake_bridge():
        os.environ["TERMINAL_CWD"] = str(proj)

    monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", fake_bridge)
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda key: None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f x"}, turn_id="turn_1", task_id="task_9"
    )
    assert rec.requests[0]["body"]["paths"] == [str(proj)]


def test_terminal_workdir_tilde_uses_subprocess_home(monkeypatch, tmp_path):
    """workdir="~/Documents" 的 `cd` 由 shell 按子进程 $HOME 展开——home_mode=
    profile 时那是 {HERMES_HOME}/home，用 Hermes 进程的 expanduser 会给真实 OS
    HOME 建快照、实际被写的 profile home 没有恢复点（Codex review P1）。"""
    import hermes_constants

    proc_home = tmp_path / "proc-home"
    sub_home = tmp_path / "profile-home"
    (proc_home / "Documents").mkdir(parents=True)
    (sub_home / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(proc_home))
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda env=None: str(sub_home))

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f a.txt", "workdir": "~/Documents"},
        turn_id="turn_1", task_id="task_9",
    )
    assert rec.requests[0]["body"]["paths"] == [str(sub_home / "Documents")]


def test_uniq_and_file_are_no_longer_readonly(monkeypatch, tmp_path):
    """uniq 的第二个位置参数是 OUTPUT（`uniq in out` 直接覆盖 out），file 的
    `-C -m` 会编译写出 .mgc：参数就能写文件的命令与配置驱动命令同理，整体移出
    只读清单（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "uniq in.txt /home/alice/Documents/notes.txt",
        "uniq notes.txt",
        "file -C -m magic",
    ):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 3, "uniq / file 要走 cwd 保护"

    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "cut -d: -f1 /etc/passwd"}, turn_id="turn_1") is None
    assert len(rec.requests) == 3, "无写入面的命令仍然只读"


def test_env_split_string_daemonizers_are_blocked(monkeypatch, tmp_path):
    """env -S/--split-string 会把字符串重新拆成命令词执行，藏在里面的
    setsid/nohup 不能当不透明参数跳过（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "env -S \"setsid -f sh -c 'sleep 1; rm -f victim'\"",
        'env -vS "setsid rm -f victim"',
        "env '-Ssetsid -f' sh -c 'rm -f victim'",
        'env --split-string="setsid rm -f victim"',
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # -S 里没有 daemonizer 的照常走保护。
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": 'env -S "sh -c" \'rm -f x\''}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "无 daemonizer 的 env -S 命令照常走保护"


def test_ssh_backend_write_commands_fail_closed(monkeypatch, tmp_path):
    """terminal.backend=ssh 的命令在远端主机执行：本机快照护不住远端文件，按
    本机路径 ensure 出来的是假恢复点；远端还可能就是设备自己（ssh 到
    loopback）。写入 fail-closed，只读命令不受影响（Codex review P1）。"""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    rec = _install(monkeypatch)

    blocked = guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f ~/Documents/a.txt"}, turn_id="turn_1"
    )
    assert blocked is not None and "NOT executed" in blocked
    assert rec.requests == [], "ssh backend 不该向本机 ensure"

    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "cat a.txt"}, turn_id="turn_1") is None
    assert rec.requests == [], "只读命令在 ssh backend 下照常放行"


def test_ssh_backend_blocks_all_write_tools(monkeypatch, tmp_path):
    """ssh backend 下 write_file / patch / execute_code 同样在远端执行
    （file_tools 按 env_type 建 SSHEnvironment、execute_code 走
    _execute_remote），本机快照护不住远端文件——全部 fail-closed
    （Codex review P1）。"""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    rec = _install(monkeypatch)
    target = tmp_path / "预算.xlsx"
    target.write_text("old")

    for tool, args in (
        ("write_file", {"path": str(target)}),
        ("patch", {"mode": "replace", "path": str(target), "search": "a", "replace": "b"}),
        ("execute_code", {"code": "open('x','w')"}),
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot(tool, args, turn_id="turn_1", task_id="task_9")
        assert blocked is not None and "NOT executed" in blocked, tool
    assert rec.requests == [], "ssh backend 不该向本机 ensure"


def test_bash_env_disables_readonly(monkeypatch, tmp_path):
    """BASH_ENV（POSIX sh 的 ENV）生效时，非交互 shell 先 source 启动脚本再跑
    命令——命令头证明不了任何事，只读放行关闭、一律按 cwd 保护
    （Codex review P1）。"""
    rec = _install(monkeypatch)
    monkeypatch.setenv("BASH_ENV", "/home/alice/.hook.sh")
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "BASH_ENV 生效时 ls 也要走 cwd 保护"

    monkeypatch.delenv("BASH_ENV", raising=False)
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "环境干净时只读放行恢复"


def test_coproc_backgrounding_is_blocked(monkeypatch, tmp_path):
    """coproc 是 bash 关键字级的后台化：协进程在命令返回后继续跑，与 nohup /
    setsid 同罪（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in ("coproc rm -f victim", "coproc W { rm -f victim; }"):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []


def test_diff_is_no_longer_readonly(monkeypatch, tmp_path):
    """GNU diff 的 `-l/--paginate` 会把输出交给 PATH 上的 `pr` 执行——参数即可
    挂外部命令，与 find -exec 同理整体移出只读清单（Codex review P1）。cmp 无
    此面，保留。"""
    rec = _install(monkeypatch)
    for cmd in ("diff -l old.txt new.txt", "diff old.txt new.txt"):
        guard.reset_for_test()
        assert guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "diff 要走 cwd 保护"

    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "cmp old.txt new.txt"}, turn_id="turn_1") is None
    assert len(rec.requests) == 2, "cmp 仍然只读"


def test_internal_parent_relative_targets_are_protected(monkeypatch, tmp_path):
    """内部 `..` 段的相对路径（sub/../../x、./../x）归一后跳出主保护目录，
    前缀正则接不住——按 shell word 归一后逃逸的整词入候选（Codex review P1）。"""
    docs = tmp_path / "Documents"
    docs.mkdir()
    target = docs / "a.txt"
    target.write_text("x")
    cwd = tmp_path / "Work"
    cwd.mkdir()
    (cwd / "sub").mkdir()

    rec = _install(monkeypatch)
    for cmd in (
        "rm -f ./../Documents/a.txt",
        "rm -f sub/../../Documents/a.txt",
    ):
        guard.reset_for_test()
        rec.requests.clear()
        guard.maybe_require_snapshot(
            "terminal", {"command": cmd, "workdir": str(cwd)},
            turn_id="turn_1", task_id="task_9",
        )
        ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
        assert str(target) in ensured, cmd

    # 主目录内的相对路径不额外加餐（cwd 快照已覆盖）。
    guard.reset_for_test()
    rec.requests.clear()
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f sub/../notes.txt", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert ensured == [str(cwd)], ensured


def test_terminal_cwd_container_fallback_maps_to_host(monkeypatch, tmp_path):
    """fallback 的 TERMINAL_CWD 本身可能是容器口径（config 把 cwd 写成
    /workspace）：与显式 workdir / session_cwd 一样要做容器反解，否则给本机
    字面 /workspace 建快照而真实被写的是 bind 的 host 目录（Codex review P1）。"""
    vol_host = tmp_path / "proj"
    vol_host.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{vol_host}:/workspace"]))
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot("terminal", {"command": "rm -f a.txt"}, turn_id="turn_1")
    assert rec.requests[0]["body"]["paths"] == [str(vol_host)]


def test_shell_init_hooks_disable_readonly(monkeypatch, tmp_path):
    """LocalEnvironment 建会话时 source ~/.profile / ~/.bash_profile /
    ~/.bashrc 并快照 alias：rc 里的 `alias ls='rm ...'` 不需要会话内定义就已
    生效，rc 文件存在且非空时关闭只读放行（Codex review P1）。"""
    rec = _install(monkeypatch)
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert rec.requests == [], "干净 HOME 下 ls 只读放行"

    (tmp_path / ".bashrc").write_text("alias ls='rm -f victim'\n")
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "存在非空 rc 时 ls 也要走 cwd 保护"


def test_custom_shell_init_files_disable_readonly(monkeypatch, tmp_path):
    """rc 遮蔽检测要复用 terminal 自己的 init 文件解析：terminal.shell_init_files
    配置的自定义文件同样会被会话 source（Codex review P1）。"""
    local_env = pytest.importorskip("tools.environments.local")
    custom = tmp_path / "init.sh"
    custom.write_text("alias ls='rm -f victim'\n")
    monkeypatch.setattr(local_env, "_resolve_shell_init_files", lambda: [str(custom)])

    rec = _install(monkeypatch)
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "自定义 init 文件生效时 ls 也要走 cwd 保护"

    monkeypatch.setattr(local_env, "_resolve_shell_init_files", lambda: [])
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1, "解析结果为空时只读放行恢复"


def test_docker_relative_workdir_maps_to_host(monkeypatch, tmp_path):
    """相对 workdir 锚在容器口径的 TERMINAL_CWD 上时，join 结果也要先容器反解
    再做 host 存在性检查，否则错退回保护 cwd（Codex review P1）。"""
    vol_host = tmp_path / "vol"
    (vol_host / "Project").mkdir(parents=True)
    (vol_host / "Documents").mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{vol_host}:/workspace"]))
    monkeypatch.setenv("TERMINAL_CWD", "/workspace/Project")
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f a.txt", "workdir": "../Documents"},
        turn_id="turn_1",
    )
    assert rec.requests[0]["body"]["paths"] == [str(vol_host / "Documents")]


def test_symlinked_relative_targets_are_protected(monkeypatch, tmp_path):
    """cwd 内的目录 symlink 指向另一个受保护目录时，`rm -f docs/a.txt` 实际
    删的是链接目标——相对词按 realpath 判逃逸后按真实目标加餐（Codex review
    P1）。"""
    docs = tmp_path / "Documents"
    docs.mkdir()
    target = docs / "a.txt"
    target.write_text("x")
    cwd = tmp_path / "Work"
    cwd.mkdir()
    os.symlink(str(docs), str(cwd / "docs"))

    rec = _install(monkeypatch)
    guard.maybe_require_snapshot(
        "terminal", {"command": "rm -f docs/a.txt", "workdir": str(cwd)},
        turn_id="turn_1", task_id="task_9",
    )
    ensured = [p for r in rec.requests for p in (r["body"].get("paths") or [])]
    assert str(target) in ensured, ensured


def test_sh_dash_c_daemonizers_are_blocked(monkeypatch, tmp_path):
    """sh/bash -c 的字面命令串会重新进入 shell 解析，与 env -S 同类：藏在里面
    的 setsid/nohup 递归判定后照样阻断（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "sh -c 'setsid -f sh -c \"sleep 1; rm -f victim\"'",
        'bash -lc "setsid rm -f victim"',
        "zsh -c 'nohup rm -f victim'",
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # 没有 daemonizer 的 sh -c 照常走保护。
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "sh -c 'rm -f x'"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1


def test_bash_dash_c_option_terminator_daemonizers_are_blocked(monkeypatch, tmp_path):
    """bash 语义里 -c 的命令串是第一个非选项操作数：`bash -c -- 'cmd'` 执行的
    是 cmd 而不是 `--`，`bash -c -l 'cmd'` 的命令串在后续选项之后——把 `--` /
    `-l` 当命令串递归会判出 False 放行（Codex review P1）。"""
    rec = _install(monkeypatch)
    for cmd in (
        "bash -c -- 'setsid -f sh -c \"sleep 1; rm -f victim\"'",
        "bash -c -l 'setsid rm -f victim'",
    ):
        guard.reset_for_test()
        blocked = guard.maybe_require_snapshot("terminal", {"command": cmd}, turn_id="turn_1")
        assert blocked is not None and "NOT executed" in blocked, cmd
    assert rec.requests == []

    # `--` 后没有 daemonizer 的照常走保护，不误伤。
    guard.reset_for_test()
    assert guard.maybe_require_snapshot(
        "terminal", {"command": "bash -c -- 'rm -f x'"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1


def test_container_backend_disables_readonly_shortcut(monkeypatch, tmp_path):
    """容器 backend 的命令在容器内 shell 执行：docker_env 注入的 BASH_ENV、镜
    像自带 ENV / rc 文件主进程都验证不了，「可证明只读」不存在——ls 也按 cwd
    拍幂等快照，不是阻断（Codex review P1）。"""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    rec = _install(monkeypatch)

    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1


def test_path_shadowed_readonly_names_are_protected(monkeypatch, tmp_path):
    """PATH 前置的同名 helper（~/bin/ls）会遮蔽系统 ls：首词必须真实解析到可
    信系统前缀才走只读快捷，否则退化按 cwd 拍快照——不是阻断
    （Codex review P1）。"""
    rec = _install(monkeypatch)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_ls = bindir / "ls"
    fake_ls.write_text("#!/bin/sh\nrm -f victim\n")
    fake_ls.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")

    assert guard.maybe_require_snapshot(
        "terminal", {"command": "ls -la"}, turn_id="turn_1") is None
    assert len(rec.requests) == 1


def test_cd_symlink_escape_targets_are_protected(monkeypatch, tmp_path):
    """`cd docs && rm a.txt` 里 docs 是指向主目录外的 symlink 时，真实目标目
    录要整目录加餐 ensure——docs 与 a.txt 都不含 `/`，逐词判定接不住
    （Codex review P1）。cd 进主目录内的普通子目录不产生加餐。"""
    rec = _install(monkeypatch)
    work = tmp_path / "Work"
    docs = tmp_path / "Documents"
    work.mkdir()
    docs.mkdir()
    (docs / "a.txt").write_text("x")
    (work / "docs").symlink_to(docs)

    assert guard.maybe_require_snapshot(
        "terminal",
        {"command": "cd docs && rm a.txt", "workdir": str(work)},
        turn_id="turn_1",
    ) is None

    all_paths = [p for r in rec.requests for p in r["body"]["paths"]]
    assert str(work) in all_paths, "主保护仍是 workdir"
    assert os.path.realpath(str(docs)) in all_paths, "cd 进的 symlink 真实目录要加餐 ensure"

    # cd 进主目录内的普通子目录：主 ensure 一次，无加餐。
    guard.reset_for_test()
    rec2 = _install(monkeypatch)
    (work / "sub").mkdir()
    assert guard.maybe_require_snapshot(
        "terminal",
        {"command": "cd sub && rm x.txt", "workdir": str(work)},
        turn_id="turn_2",
    ) is None
    assert len(rec2.requests) == 1
    assert rec2.requests[0]["body"]["paths"] == [str(work)]


def test_registry_dispatch_direct_path_is_gated(monkeypatch, tmp_path):
    """插件公开 API ctx.dispatch_tool() 直连 registry.dispatch()、不经
    handle_function_call——gate 在 registry 统一分发入口必须同样生效
    （Codex review P1）。"""
    from tools.registry import registry

    _install(monkeypatch, {"ready": False, "operations": []})
    target = tmp_path / "a.txt"
    target.write_text("x")

    out = registry.dispatch(
        "write_file", {"path": str(target), "content": "y"}, turn_id="turn_r1")
    assert "error" in json.loads(out)
    assert target.read_text() == "x", "gate 先于 handler 执行"

    # 没有 turn 上下文的直连调用在设备上同样 fail-closed（missing turn id）。
    guard.reset_for_test()
    _install(monkeypatch, {"ready": True, "operations": []})
    out2 = registry.dispatch("write_file", {"path": str(target), "content": "y"})
    assert "error" in json.loads(out2)
    assert target.read_text() == "x"
