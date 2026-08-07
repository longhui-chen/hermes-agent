import json

from agent import secret_scope
from tools import terminal_tool as terminal_tool_module


def test_managed_lark_cli_runs_through_process_authenticated_broker(monkeypatch):
    captured = {}

    class Result:
        output = '{"items":[{"subject":"hello"}]}'
        exit_code = 0
        timed_out = False

    def fake_request(agent_id, args, *, timeout_seconds):
        captured.update(
            agent_id=agent_id,
            args=args,
            timeout_seconds=timeout_seconds,
        )
        return Result()

    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        "agent.credential_broker.request_lark_cli",
        fake_request,
    )
    token = secret_scope.set_secret_scope({"ZET_AGENT_ID": "agent-1"})
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        raw = terminal_tool_module._run_lark_cli_command_if_allowed(
            "lark-cli mail user_mailbox.messages list "
            "--user-mailbox-id me --jq '.items[0].subject'",
            timeout=45,
        )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(token)

    result = json.loads(raw)
    assert captured == {
        "agent_id": "agent-1",
        "args": [
            "mail",
            "user_mailbox.messages",
            "list",
            "--user-mailbox-id",
            "me",
            "--jq",
            ".items[0].subject",
        ],
        "timeout_seconds": 45,
    }
    assert result["output"] == Result.output
    assert result["exit_code"] == 0
    assert result["lark_cli_brokered"] is True


def test_terminal_flow_dispatches_authorized_lark_cli_without_local_shell(
    monkeypatch,
):
    captured = {}

    class Result:
        output = '{"authenticated":true}'
        exit_code = 0
        timed_out = False

    def fake_request(agent_id, args, *, timeout_seconds):
        captured.update(
            agent_id=agent_id,
            args=args,
            timeout_seconds=timeout_seconds,
        )
        return Result()

    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "agent.credential_broker.request_lark_cli",
        fake_request,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *args, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        "tools.approval.get_current_session_key",
        lambda default="": default,
    )
    token = secret_scope.set_secret_scope({"ZET_AGENT_ID": "agent-1"})
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        raw = terminal_tool_module.terminal_tool(
            "lark-cli auth status",
            task_id="lark-cli-broker-flow",
            timeout=40,
        )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(token)

    result = json.loads(raw)
    assert captured == {
        "agent_id": "agent-1",
        "args": ["auth", "status"],
        "timeout_seconds": 40,
    }
    assert result["output"] == Result.output
    assert result["exit_code"] == 0
    assert result["lark_cli_brokered"] is True


def test_managed_lark_cli_compound_or_background_commands_fail_closed(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")

    direct = terminal_tool_module._run_lark_cli_command_if_allowed(
        "lark-cli whoami; env",
        timeout=30,
    )
    guarded = terminal_tool_module._lark_cli_shell_guard_result(
        "lark-cli whoami",
    )

    assert json.loads(direct)["errorCode"] == "lark_cli_compound_command"
    assert json.loads(guarded)["errorCode"] == "lark_cli_direct_only"
    assert json.loads(
        terminal_tool_module._run_lark_cli_command_if_allowed(
            "timeout 30 lark-cli whoami",
            timeout=30,
        )
    )["errorCode"] == "lark_cli_compound_command"
    assert json.loads(
        terminal_tool_module._run_lark_cli_command_if_allowed(
            "/usr/local/bin/lark-cli whoami",
            timeout=30,
        )
    )["errorCode"] == "lark_cli_compound_command"
    assert terminal_tool_module._lark_cli_shell_guard_result(
        "echo lark-cli"
    ) is None


def test_unmanaged_lark_cli_keeps_normal_local_shell_behavior(monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED_GATEWAY", raising=False)

    assert terminal_tool_module._run_lark_cli_command_if_allowed(
        "lark-cli whoami",
        timeout=30,
    ) is None
    assert terminal_tool_module._lark_cli_shell_guard_result(
        "lark-cli whoami"
    ) is None
