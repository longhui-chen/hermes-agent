import json
import os
import shlex
import textwrap
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import secret_scope
from gateway.session_context import set_zettlab_turn_id, zettlab_turn_id
from tools import terminal_tool as terminal_tool_module


@pytest.fixture(autouse=True)
def _reset_agent_creator_runtime(monkeypatch):
    previous_multiplex = secret_scope.is_multiplex_active()
    previous_turn_id = zettlab_turn_id()
    scope_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    set_zettlab_turn_id("")
    monkeypatch.setattr(
        terminal_tool_module,
        "_CONNECTOR_RUNTIME_ROOT_ANCHOR",
        None,
    )
    yield
    set_zettlab_turn_id(previous_turn_id)
    secret_scope.set_multiplex_active(previous_multiplex)
    secret_scope.reset_secret_scope(scope_token)


@contextmanager
def _scope(values):
    token = secret_scope.set_secret_scope(values)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)


def _write_creator(
    tmp_path,
    body: str,
    *,
    runtime_capabilities: list[str] | None = None,
) -> os.PathLike:
    script = (
        tmp_path
        / "presets"
        / "skills"
        / "agent-creator"
        / "scripts"
        / "create_agent.py"
    )
    script.parent.mkdir(parents=True)
    preamble = """
import os as _secret_os

def _read_injected_secret(key):
    descriptor = int(_secret_os.environ.pop(key + "_FD"))
    with _secret_os.fdopen(descriptor, "rb", closefd=True) as stream:
        return stream.read(4097).decode("utf-8")

"""
    script.write_text(
        textwrap.dedent(preamble).lstrip() + textwrap.dedent(body).lstrip(),
        encoding="utf-8",
    )
    capabilities = runtime_capabilities
    if capabilities is None:
        capabilities = ["zettlab.agent_action_token_fd.v1"]
    manifest = script.parent.parent / "manifest.yaml"
    manifest.write_text(
        "id: agent-creator\n"
        "runtime_capabilities:\n"
        + "".join(f"  - {capability}\n" for capability in capabilities),
        encoding="utf-8",
    )
    return script


def _configure(monkeypatch, tmp_path, body: str):
    script = _write_creator(tmp_path, body)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    return script


def _canonical_command(suffix: str) -> str:
    return (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
        f'scripts/create_agent.py" {suffix}'
    )


def test_terminal_dispatch_runs_preflight_with_only_scoped_credentials(
    monkeypatch,
    tmp_path,
):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys

        print(json.dumps({
            "argv": sys.argv[1:],
            "isolated": sys.flags.isolated,
            "no_site": sys.flags.no_site,
            "site_packages_visible": any(
                "site-packages" in entry or "dist-packages" in entry
                for entry in sys.path
            ),
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == "scope-token"
            ),
            "token_env_absent": "ZETTLAB_AGENT_ACTION_TOKEN" not in os.environ,
            "turn_ok": os.environ.get("ZETTLAB_TURN_ID") == "turn-current",
        }))
        """,
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "wrong-global-token")
    monkeypatch.setenv("ZETTLAB_TURN_ID", "wrong-global-turn")
    set_zettlab_turn_id("turn-current")

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("preflight"),
                task_id="agent-creator-preflight-flow",
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["exit_code"] == 0
    output = json.loads(result["output"])
    assert output == {
        "argv": ["preflight"],
        "isolated": 1,
        "no_site": 1,
        "site_packages_visible": False,
        "token_ok": True,
        "token_env_absent": True,
        "turn_ok": True,
    }
    assert "scope-token" not in result["output"]
    assert "wrong-global-token" not in result["output"]


def test_terminal_dispatch_runs_fixed_read_only_list(monkeypatch, tmp_path):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import sys

        print(json.dumps({
            "argv": sys.argv[1:],
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == "scope-token"
            ),
        }))
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("list"),
                task_id="agent-creator-list-flow",
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "argv": ["list"],
        "token_ok": True,
    }


@pytest.mark.skipif(
    os.name == "nt",
    reason="Agent Creator secret FD transport is POSIX-only",
)
@pytest.mark.parametrize("command_name", ["preflight", "list"])
def test_old_presets_capability_mismatch_blocks_before_secret_or_popen(
    monkeypatch,
    tmp_path,
    command_name,
):
    script = _configure(
        monkeypatch,
        tmp_path,
        """
        import os
        import sys

        if not os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN"):
            print("legacy preset plaintext credential unavailable", file=sys.stderr)
            raise SystemExit(78)
        print("must not run with plaintext credential")
        """,
    )
    (script.parent.parent / "manifest.yaml").write_text(
        "id: agent-creator\nruntime_capabilities: []\n",
        encoding="utf-8",
    )
    from tools import trusted_direct_runner
    from tools.environments import local as local_environment

    real_popen = trusted_direct_runner.subprocess.Popen

    def forbidden(*_args, **_kwargs):
        raise AssertionError("capability mismatch must not acquire a token or spawn")

    monkeypatch.setattr(
        local_environment,
        "build_agent_creator_runtime_env",
        forbidden,
    )
    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", forbidden)

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command(command_name),
                task_id=f"agent-creator-{command_name}-presets-mismatch",
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["agent_creator_blocked"] is True
    assert result["exit_code"] == 2
    assert (
        result["errorCode"]
        == "agent_creator_runtime_capability_unavailable"
    )
    assert "scope-token" not in result["output"]
    assert "must not run" not in result["output"]

    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", real_popen)
    healthy = json.loads(
        terminal_tool_module.terminal_tool(
            "printf service-healthy",
            task_id="agent-creator-capability-mismatch-other-terminal",
        )
    )
    assert healthy["exit_code"] == 0
    assert healthy["output"] == "service-healthy"


@pytest.mark.parametrize(
    "manifest_text",
    [
        None,
        "id: agent-creator\nruntime_capabilities: malformed\n",
        "id: agent-creator\nruntime_capabilities: [\n",
        (
            "id: agent-creator\nruntime_capabilities:\n"
            + "".join(f"  - capability-{index}\n" for index in range(33))
        ),
        "x" * (64 * 1024 + 1),
    ],
    ids=["missing", "not_a_list", "invalid_yaml", "too_many", "oversized"],
)
def test_manifest_capability_parse_fails_before_secret_or_popen(
    monkeypatch,
    tmp_path,
    manifest_text,
):
    script = _configure(monkeypatch, tmp_path, "print('must not run')\n")
    manifest = script.parent.parent / "manifest.yaml"
    if manifest_text is None:
        manifest.unlink()
    else:
        manifest.write_text(manifest_text, encoding="utf-8")
    from tools import trusted_direct_runner
    from tools.environments import local as local_environment

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid manifest must not acquire a token or spawn")

    monkeypatch.setattr(
        local_environment,
        "build_agent_creator_runtime_env",
        forbidden,
    )
    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", forbidden)

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_runtime_capability_unavailable"
    assert "must not run" not in json.dumps(result)


def test_model_tools_registry_dispatch_reaches_agent_creator_worker(
    monkeypatch,
    tmp_path,
):
    import model_tools

    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os

        print(json.dumps({
            "worker": "agent-creator",
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == "scope-token"
            ),
        }))
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            model_tools.handle_function_call(
                "terminal",
                {"command": _canonical_command("preflight")},
                task_id="agent-creator-registry-dispatch",
                skip_pre_tool_call_hook=True,
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "worker": "agent-creator",
        "token_ok": True,
    }


@pytest.mark.skipif(
    os.name == "nt",
    reason="Fixed packaged CLI flow uses a POSIX test executable",
)
def test_canonical_preset_shim_can_call_fixed_agentcomputer_cli(
    monkeypatch,
    tmp_path,
):
    cli = tmp_path / "agentcomputer"
    cli.write_text(
        """#!/bin/sh
IFS= read -r token
if [ "$token" != "scope-token" ]; then
  exit 9
fi
printf '{"argv":["%s","%s","%s"],"token_ok":true}\\n' "$1" "$2" "$3"
""",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    _configure(
        monkeypatch,
        tmp_path,
        f"""
        import subprocess

        result = subprocess.run(
            [{str(cli)!r}, "agent", "preflight", "--json"],
            capture_output=True,
            check=False,
            input=_read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN") + "\\n",
            text=True,
        )
        print(result.stdout, end="")
        raise SystemExit(result.returncode)
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("preflight"),
                task_id="agent-creator-fixed-cli-flow",
            )
        )

    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "argv": ["agent", "preflight", "--json"],
        "token_ok": True,
    }


def test_agentcomputer_mutation_requires_one_shot_approval_before_token(
    monkeypatch,
    tmp_path,
):
    _configure(monkeypatch, tmp_path, "raise SystemExit('must not run')\n")
    captured = {}

    def request_approval(tool_name, reason, **kwargs):
        captured.update(tool_name=tool_name, reason=reason, kwargs=kwargs)
        return {
            "approved": False,
            "status": "pending_approval",
            "description": reason,
            "pattern_key": "agentcomputer:file.delete",
        }

    monkeypatch.setattr("tools.approval.request_tool_approval", request_approval)
    monkeypatch.setattr(
        "tools.environments.local.build_agent_creator_runtime_env",
        lambda: (_ for _ in ()).throw(AssertionError("token acquired too early")),
    )

    result = json.loads(terminal_tool_module.terminal_tool(
        _canonical_command("cli file delete --path notes/a.txt"),
        task_id="agentcomputer-delete-approval",
    ))

    assert result["status"] == "pending_approval"
    assert result["approval_pending"] is True
    assert captured["tool_name"] == "agentcomputer_cli"
    assert captured["kwargs"]["one_shot"] is True
    assert captured["kwargs"]["allow_yolo_bypass"] is False


def test_agentcomputer_read_only_cli_does_not_request_mutation_approval(
    monkeypatch,
    tmp_path,
):
    _configure(monkeypatch, tmp_path, "print('read-only-ok')\n")
    monkeypatch.setattr(
        "tools.approval.request_tool_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("read-only CLI must not request mutation approval")
        ),
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(terminal_tool_module.terminal_tool(
            _canonical_command("cli file list --path . --limit 20"),
            task_id="agentcomputer-list-no-approval",
        ))

    assert result["exit_code"] == 0
    assert result["output"].strip() == "read-only-ok"


def test_concurrent_creator_calls_keep_profile_scope_and_turn_isolated(
    monkeypatch,
    tmp_path,
):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys
        import time

        name = json.loads(sys.argv[3])["name"]
        time.sleep(0.1)
        print(json.dumps({
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == f"token-{name}"
            ),
            "turn_ok": os.environ.get("ZETTLAB_TURN_ID") == f"turn-{name}",
        }))
        """,
    )
    terminal_tool_module._capture_connector_runtime_root()

    def run(name: str) -> dict:
        payload = shlex.quote(json.dumps({"name": name}))
        previous_turn = zettlab_turn_id()
        set_zettlab_turn_id(f"turn-{name}")
        try:
            with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": f"token-{name}"}):
                return json.loads(
                    terminal_tool_module._run_agent_creator_command_if_allowed(
                        _canonical_command(f"create --payload {payload}"),
                        cwd=str(tmp_path),
                        timeout=5,
                    )
                )
        finally:
            set_zettlab_turn_id(previous_turn)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("alpha", "beta")))

    assert [result["exit_code"] for result in results] == [0, 0]
    assert [json.loads(result["output"]) for result in results] == [
        {"token_ok": True, "turn_ok": True},
        {"token_ok": True, "turn_ok": True},
    ]
    assert all(
        secret not in json.dumps(results)
        for secret in ("token-alpha", "token-beta", "turn-alpha", "turn-beta")
    )


def test_terminal_dispatch_passes_bounded_create_payload_over_stdin(
    monkeypatch,
    tmp_path,
):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys

        print(json.dumps({
            "argv": sys.argv[1:],
            "payload": json.loads(sys.stdin.read()),
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == "scope-token"
            ),
        }, ensure_ascii=False))
        """,
    )
    payload = {
        "name": "销售助手",
        "soul_identity": "帮助用户推进企业客户",
        "memory_entries": [{"title": "框架", "body": "使用 SPIN"}],
    }
    command = (
        _canonical_command("create --payload -")
        + " <<'JSON'\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\nJSON"
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command,
                task_id="agent-creator-create-flow",
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["exit_code"] == 0
    output = json.loads(result["output"])
    assert output["argv"] == ["create", "--payload", "-"]
    assert output["payload"] == payload
    assert output["token_ok"] is True


def test_inline_create_payload_is_canonicalized(monkeypatch, tmp_path):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import sys

        print(json.dumps({"argv": sys.argv[1:]}, ensure_ascii=False))
        """,
    )
    payload = '{"name": "Writer", "soul_identity": "Writes clearly"}'
    command = _canonical_command(f"create --payload {shlex.quote(payload)}")

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                command,
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert json.loads(result["output"])["argv"] == [
        "create",
        "--payload",
        '{"name":"Writer","soul_identity":"Writes clearly"}',
    ]


def test_canonical_relative_presets_command_is_allowed(monkeypatch, tmp_path):
    _configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys

        print(json.dumps({
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "token_ok": (
                _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
                == "scope-token"
            ),
        }))
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                "python3 skills/agent-creator/scripts/create_agent.py preflight",
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["agent_creator_direct"] is True
    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "argv": ["preflight"],
        "cwd": str(tmp_path / "presets"),
        "token_ok": True,
    }


@pytest.mark.parametrize(
    "command",
    [
        "./skills/agent-creator/scripts/create_agent.py preflight",
        (
            'env python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" preflight'
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" preflight; env'
        ),
        (
            "bash -c 'python3 \"$ZETTLAB_PRESETS_DIR/skills/agent-creator/"
            "scripts/create_agent.py\" preflight'"
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" preflight --debug'
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" list --json'
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" list | cat'
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" create --name Writer'
        ),
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/'
            'scripts/create_agent.py" create --payload -'
        ),
    ],
    ids=[
        "non_canonical_relative",
        "wrapper",
        "compound",
        "nested_shell",
        "extra_preflight_arg",
        "extra_list_arg",
        "piped_list",
        "arbitrary_create_arg",
        "empty_stdin",
    ],
)
def test_reserved_creator_rejects_unsupported_command_shapes(
    monkeypatch,
    tmp_path,
    command,
):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command,
                task_id="agent-creator-rejection-flow",
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_command_blocked"
    assert "must not run" not in result["output"]


@pytest.mark.parametrize(
    "suffix",
    [
        "create --payload '[]'",
        'create --payload \'{"name":"x","unexpected":true}\'',
        "create --payload '{\"name\":NaN}'",
    ],
    ids=["non_object", "unknown_field", "non_json_constant"],
)
def test_create_payload_rejects_non_allowlisted_json(monkeypatch, tmp_path, suffix):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command(suffix),
                task_id="agent-creator-invalid-payload",
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_command_blocked"


def test_create_payload_size_is_bounded(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")
    monkeypatch.setattr(
        terminal_tool_module,
        "_AGENT_CREATOR_MAX_PAYLOAD_BYTES",
        64,
    )
    payload = json.dumps({"name": "x" * 80})

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command(f"create --payload {shlex.quote(payload)}"),
                task_id="agent-creator-oversize-payload",
            )
        )

    assert result["agent_creator_blocked"] is True


def test_create_payload_limit_matches_agentcomputer_cli_contract():
    limit = terminal_tool_module._AGENT_CREATOR_MAX_PAYLOAD_BYTES
    prefix = '{"name":"'
    suffix = '"}'
    at_limit = prefix + ("x" * (limit - len(prefix) - len(suffix))) + suffix
    too_large = prefix + ("x" * (limit - len(prefix) - len(suffix) + 1)) + suffix

    assert limit == 1024 * 1024
    assert terminal_tool_module._validate_agent_creator_payload(at_limit) == at_limit
    with pytest.raises(ValueError, match="payload too large"):
        terminal_tool_module._validate_agent_creator_payload(too_large)


@pytest.mark.parametrize(
    "command",
    [
        "echo create_agent.py",
        "git diff -- create_agent.py",
        "echo 'python3 create_agent.py preflight'",
    ],
    ids=["echo-name", "git-diff-path", "quoted-command-text"],
)
def test_non_executing_creator_mentions_are_not_reserved(command):
    assert terminal_tool_module._agent_creator_shell_guard_result(command) is None


def test_non_executing_creator_mention_reaches_generic_terminal(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    task_id = "agent-creator-non-executing-mention"
    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(
                "printf '%s\\n' create_agent.py",
                task_id=task_id,
            )
        )
    finally:
        terminal_tool_module.cleanup_vm(task_id)

    assert result["exit_code"] == 0
    assert result["output"].strip() == "create_agent.py"
    assert "agent_creator_blocked" not in result


@pytest.mark.parametrize("scope", [None, {}], ids=["no_scope", "missing_token"])
def test_agent_creator_fails_closed_without_scoped_token(
    monkeypatch,
    tmp_path,
    scope,
):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "global-token-must-not-win")

    with _scope(scope):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("preflight"),
                task_id="agent-creator-no-scope",
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_scope_unavailable"
    assert "global-token-must-not-win" not in json.dumps(result)


def test_agent_creator_token_is_out_of_outer_process_env(monkeypatch, tmp_path):
    from tools import trusted_direct_runner

    _configure(monkeypatch, tmp_path, "print('unused')\n")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "stale-global-token")
    set_zettlab_turn_id("turn-current")
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return trusted_direct_runner.TrustedPythonResult(output="ok", returncode=0)

    monkeypatch.setattr(
        trusted_direct_runner,
        "run_trusted_python_script",
        fake_run,
    )
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in captured["base_env"]
    assert "ZETTLAB_TURN_ID" not in captured["base_env"]
    assert captured["injected_env"] == {"ZETTLAB_TURN_ID": "turn-current"}
    assert captured["injected_secrets"] == {
        "ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"
    }
    assert captured["script_bytes"] == captured["script"].read_bytes()
    assert captured["stdlib_only"] is True


def test_agent_creator_token_is_not_in_popen_env_or_argv(monkeypatch, tmp_path):
    from tools import trusted_direct_runner

    _configure(monkeypatch, tmp_path, "print('ok')\n")
    real_popen = trusted_direct_runner.subprocess.Popen
    captured = {}

    def capture_popen(*args, **kwargs):
        captured["argv"] = args[0]
        captured["env"] = kwargs.get("env", {})
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", capture_popen)
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert "scope-token" not in json.dumps(captured)
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in captured["env"]


def test_generic_terminal_never_receives_agent_creator_token(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "stale-global-token")
    task_id = "agent-creator-generic-token-isolation"
    try:
        with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
            result = json.loads(
                terminal_tool_module.terminal_tool(
                    (
                        "python3 -c 'import os; "
                        'print(os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "absent"))\''
                    ),
                    task_id=task_id,
                )
            )
    finally:
        terminal_tool_module.cleanup_vm(task_id)

    assert result["exit_code"] == 0
    assert result["output"].strip() == "absent"
    assert "scope-token" not in json.dumps(result)
    assert "stale-global-token" not in json.dumps(result)


def test_generic_background_and_helper_envs_scrub_agent_creator_token(monkeypatch):
    from tools.environments.local import (
        _make_run_env,
        _sanitize_subprocess_env,
        hermes_subprocess_env,
    )

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "stale-global-token")
    supplied = {"ZETTLAB_AGENT_ACTION_TOKEN": "caller-token"}

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        envs = (
            _make_run_env(supplied),
            _sanitize_subprocess_env(supplied),
            hermes_subprocess_env(),
        )

    for env in envs:
        assert "ZETTLAB_AGENT_ACTION_TOKEN" not in env


def test_canonical_absolute_path_is_allowed(monkeypatch, tmp_path):
    script = _configure(monkeypatch, tmp_path, "print('absolute-ok')\n")
    command = f"{shlex.quote(os.sys.executable)} {shlex.quote(str(script))} preflight"

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command,
                task_id="agent-creator-absolute",
            )
        )

    assert result["exit_code"] == 0
    assert result["output"] == "absolute-ok"


@pytest.mark.skipif(
    os.name == "nt",
    reason="Creating symlinks requires elevated privileges on Windows",
)
def test_symlink_and_path_escape_are_rejected(monkeypatch, tmp_path):
    script = _configure(monkeypatch, tmp_path, "print('must not run')\n")
    real_script = script.with_name("real_create_agent.py")
    script.rename(real_script)
    script.symlink_to(real_script.name)
    escaped = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/agent-creator/scripts/'
        '../scripts/create_agent.py" preflight'
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        symlink_result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("preflight"),
                task_id="agent-creator-symlink",
            )
        )
        escape_result = json.loads(
            terminal_tool_module.terminal_tool(
                escaped,
                task_id="agent-creator-path-escape",
            )
        )

    assert symlink_result["agent_creator_blocked"] is True
    assert escape_result["agent_creator_blocked"] is True
    assert "must not run" not in json.dumps((symlink_result, escape_result))


def test_identity_replacement_is_rejected_before_exec(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")
    command = _canonical_command("preflight")
    parsed = terminal_tool_module._parse_agent_creator_command(command)
    assert parsed is not None
    monkeypatch.setattr(
        terminal_tool_module,
        "_parse_agent_creator_command",
        lambda _command: parsed,
    )
    presets = tmp_path / "presets"
    presets.rename(tmp_path / "presets-old")
    presets.mkdir()

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                command,
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_identity_changed"
    assert "must not run" not in json.dumps(result)


def test_oversize_verified_source_is_rejected(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")
    monkeypatch.setattr(
        terminal_tool_module,
        "_AGENT_CREATOR_MAX_SCRIPT_BYTES",
        8,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_identity_changed"
    assert "must not run" not in json.dumps(result)


def test_verified_source_is_frozen_before_worker_launch(monkeypatch, tmp_path):
    from tools import trusted_direct_runner

    script = _configure(monkeypatch, tmp_path, "print('verified-source')\n")
    original_run = trusted_direct_runner.run_trusted_python_script

    def replace_path_then_run(**kwargs):
        script.write_text("print('replaced-source')\n", encoding="utf-8")
        return original_run(**kwargs)

    monkeypatch.setattr(
        trusted_direct_runner,
        "run_trusted_python_script",
        replace_path_then_run,
    )
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert result["output"] == "verified-source"
    assert "replaced-source" not in result["output"]


@pytest.mark.parametrize(
    "mode_kwargs",
    [{"background": True}, {"pty": True}],
    ids=["background", "pty"],
)
def test_agent_creator_rejects_non_foreground_modes(
    monkeypatch,
    tmp_path,
    mode_kwargs,
):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                _canonical_command("preflight"),
                task_id="agent-creator-mode-rejection",
                **mode_kwargs,
            )
        )

    assert result["agent_creator_blocked"] is True
    assert "must not run" not in json.dumps(result)


def test_agent_creator_output_is_redacted_and_bounded(monkeypatch, tmp_path):
    from tools import tool_output_limits

    _configure(
        monkeypatch,
        tmp_path,
        """
        import os

        token = _read_injected_secret("ZETTLAB_AGENT_ACTION_TOKEN")
        print(token)
        print("x" * 10000)
        print(token)
        """,
    )
    monkeypatch.setattr(tool_output_limits, "get_max_bytes", lambda: 512)

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert len(result["output"]) <= 512
    assert "scope-token" not in result["output"]
    assert "[REDACTED]" in result["output"]
    assert "OUTPUT TRUNCATED" in result["output"]


def test_agent_creator_output_redacts_secret_split_by_ansi(monkeypatch, tmp_path):
    _configure(
        monkeypatch,
        tmp_path,
        """
        print("scope-\\x1b[31mtoken\\x1b[0m")
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 0
    assert "scope-token" not in result["output"]
    assert result["output"] == "[REDACTED]"


def test_streaming_redactor_hides_secret_split_across_chunks():
    from tools import trusted_direct_runner

    secret = "scope-" + ("x" * 128) + "-token"
    sink = trusted_direct_runner._BoundedText(256)
    redactor = trusted_direct_runner._StreamingSecretRedactor([secret], sink)
    redactor.feed("prefix:" + secret[:71])
    redactor.feed(secret[71:] + ":suffix")
    redactor.feed("", final=True)

    output = sink.render()
    assert secret not in output
    assert "scope-" not in output
    assert "-token" not in output
    assert output == "prefix:[REDACTED]:suffix"


@pytest.mark.parametrize(
    ("scope", "turn_id"),
    [
        ({"ZETTLAB_AGENT_ACTION_TOKEN": "x" * 4097}, ""),
        ({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}, "t" * 257),
    ],
    ids=["oversize_token", "oversize_turn_id"],
)
def test_agent_creator_rejects_oversize_scoped_values(
    monkeypatch,
    tmp_path,
    scope,
    turn_id,
):
    _configure(monkeypatch, tmp_path, "print('must not run')\n")
    set_zettlab_turn_id(turn_id)

    with _scope(scope):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["agent_creator_blocked"] is True
    assert result["errorCode"] == "agent_creator_scope_unavailable"
    assert "must not run" not in json.dumps(result)


@pytest.mark.live_system_guard_bypass
def test_agent_creator_timeout_kills_descendant_process_tree(monkeypatch, tmp_path):
    marker = tmp_path / "escaped-child"
    _configure(
        monkeypatch,
        tmp_path,
        f"""
        import subprocess
        import sys
        import time

        subprocess.Popen([
            sys.executable,
            "-c",
            "import pathlib,time; time.sleep(0.8); "
            "pathlib.Path({str(marker)!r}).write_text('escaped')",
        ])
        time.sleep(30)
        """,
    )

    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=0.2,
            )
        )

    assert result["exit_code"] == 124
    assert "timed out" in result["error"]
    time.sleep(1.0)
    assert not marker.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX process-group regression",
)
@pytest.mark.live_system_guard_bypass
def test_timeout_kills_stdout_holder_after_leader_exits(monkeypatch, tmp_path):
    marker = tmp_path / "stdout-holder-escaped"
    _configure(
        monkeypatch,
        tmp_path,
        f"""
        import subprocess
        import sys

        subprocess.Popen([
            sys.executable,
            "-c",
            "import pathlib,time; time.sleep(3); "
            "pathlib.Path({str(marker)!r}).write_text('escaped')",
        ])
        """,
    )

    started = time.monotonic()
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=0.2,
            )
        )
    elapsed = time.monotonic() - started

    assert result["exit_code"] == 124
    assert elapsed < 2.5
    time.sleep(1.0)
    assert not marker.exists()


@pytest.mark.live_system_guard_bypass
def test_agent_creator_interrupt_kills_descendant_process_tree(monkeypatch, tmp_path):
    from tools import interrupt

    marker = tmp_path / "interrupt-child-escaped"
    _configure(
        monkeypatch,
        tmp_path,
        f"""
        import subprocess
        import sys
        import time

        subprocess.Popen([
            sys.executable,
            "-c",
            "import pathlib,time; time.sleep(0.8); "
            "pathlib.Path({str(marker)!r}).write_text('escaped')",
        ])
        time.sleep(30)
        """,
    )
    checks = 0

    def interrupt_after_worker_starts():
        nonlocal checks
        checks += 1
        return checks >= 3

    monkeypatch.setattr(interrupt, "is_interrupted", interrupt_after_worker_starts)
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=5,
            )
        )

    assert result["exit_code"] == 130
    assert "interrupted" in result["error"]
    time.sleep(1.0)
    assert not marker.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX process-group regression",
)
@pytest.mark.live_system_guard_bypass
def test_timeout_kills_child_spawned_by_sigterm_handler(monkeypatch, tmp_path):
    marker = tmp_path / "late-child-escaped"
    _configure(
        monkeypatch,
        tmp_path,
        f"""
        import signal
        import subprocess
        import sys
        import time

        def on_term(_signum, _frame):
            subprocess.Popen([
                sys.executable,
                "-c",
                "import pathlib,time; time.sleep(3); "
                "pathlib.Path({str(marker)!r}).write_text('escaped')",
            ])
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, on_term)
        time.sleep(30)
        """,
    )

    started = time.monotonic()
    with _scope({"ZETTLAB_AGENT_ACTION_TOKEN": "scope-token"}):
        result = json.loads(
            terminal_tool_module._run_agent_creator_command_if_allowed(
                _canonical_command("preflight"),
                cwd=str(tmp_path),
                timeout=0.2,
            )
        )
    elapsed = time.monotonic() - started

    assert result["exit_code"] == 124
    assert elapsed < 2.5
    time.sleep(1.0)
    assert not marker.exists()
