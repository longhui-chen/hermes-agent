import json
from pathlib import Path

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from cron.execution_context import execution_scope
from tools import terminal_tool as terminal
from tools.environments.local import build_camera_semantic_runtime_env


SCRIPT = "skills/camera-semantic-evaluation/scripts/camera_semantic_monitor.py"
COMMAND = f'python3 "$ZETTLAB_PRESETS_DIR/{SCRIPT}" candidate --policy-id 12345678-1234-1234-1234-123456789abc'


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    root = tmp_path / "presets"
    script = root / SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text(
        "import os,json\n"
        "def read(name):\n"
        "    return os.read(int(os.environ[name+'_FD']),4096).decode()\n"
        "print(json.dumps({'job_ok':read('ZETTLAB_CAMERA_JOB_ID')=='job-a',"
        "'run_ok':read('ZETTLAB_CAMERA_EXECUTION_ID')=='run-a',"
        "'secret':read('ZETTLAB_AGENT_ACTION_TOKEN'),"
        "'plain':os.environ.get('ZETTLAB_AGENT_ACTION_TOKEN',''),"
        "'ambient':os.environ.get('UNRELATED_SECRET',''),"
        "'url':os.environ.get('ZETTLAB_LOCAL_SERVER_URL','')}))\n"
    )
    (script.parent.parent / "manifest.yaml").write_text(
        "id: camera-semantic-evaluation\nrequired_scopes: [hardware.camera:read]\n"
        "runtime_capabilities: [zettlab.camera.semantic.v1]\n"
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-inherit")
    monkeypatch.setattr(terminal, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(terminal, "_connector_runtime_path_is_trusted", lambda *a, **k: True)
    monkeypatch.setattr(terminal, "_ensure_sensitive_runtime_boundary", lambda: True)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    token = set_secret_scope({
        "ZETTLAB_AGENT_ACTION_TOKEN": "a" * 64,
        "ZET_AGENT_OUTPUT_DIR": str(tmp_path / "output"),
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:19090/api/v1/internal/chat/append",
    })
    try:
        yield tmp_path, script
    finally:
        reset_secret_scope(token)


def test_terminal_semantic_flow_uses_private_fds(runtime):
    profile, _ = runtime
    with execution_scope("job-a", "run-a", profile):
        result = json.loads(terminal.terminal_tool(command=COMMAND, task_id="semantic-fd-flow"))
    assert result["camera_runtime_direct"] is True
    assert result["exit_code"] == 0
    payload = json.loads(result["output"])
    assert payload.pop("secret") in {"[REDACTED]", "***"}
    assert "a" * 64 not in result["output"]
    assert payload == {
        "job_ok": True, "run_ok": True,
        "plain": "", "ambient": "", "url": "http://127.0.0.1:19090",
    }


@pytest.mark.parametrize("scope_kind", ["missing", "wrong-profile"])
def test_runner_rejects_missing_or_mismatched_task(runtime, monkeypatch, scope_kind):
    profile, _ = runtime
    monkeypatch.setenv("ZETTLAB_CAMERA_JOB_ID", "job-a")
    monkeypatch.setenv("ZETTLAB_CAMERA_EXECUTION_ID", "run-a")
    if scope_kind == "missing":
        result = terminal._run_camera_runtime_command_if_allowed(COMMAND, cwd=str(profile), timeout=5)
    else:
        with execution_scope("job-a", "run-a", Path("/wrong-profile")):
            result = terminal._run_camera_runtime_command_if_allowed(COMMAND, cwd=str(profile), timeout=5)
    assert json.loads(result)["exit_code"] == -1


@pytest.mark.parametrize("suffix", ["; id", " --job-id other", " --output /tmp/leak", " --policy-id duplicate"])
def test_parser_rejects_scope_and_shell_overrides(runtime, suffix):
    assert terminal._parse_camera_runtime_command(COMMAND + suffix) is None
    assert json.loads(terminal._run_camera_runtime_command_if_allowed(
        COMMAND + suffix, cwd=str(runtime[0]), timeout=5
    ))["camera_runtime_blocked"] is True


def test_candidate_and_commit_have_separate_parameter_allowlists(runtime):
    assert terminal._parse_camera_runtime_command(COMMAND) is not None
    prefix = f'python3 "$ZETTLAB_PRESETS_DIR/{SCRIPT}" commit --capability ' + "a" * 32
    assert terminal._parse_camera_runtime_command(prefix + " --matched false") is not None
    positive = prefix + " --matched true --subject-kind person --predicate lingers --duration-seconds 10 --evidence-ref window-frame-1.jpg"
    assert terminal._parse_camera_runtime_command(positive) is not None
    assert terminal._parse_camera_runtime_command(positive.replace("seconds 10", "seconds 3601")) is None
    assert terminal._parse_camera_runtime_command(positive + " --matched false") is None


def test_changed_package_is_rejected(runtime):
    profile, script = runtime
    assert terminal._parse_camera_runtime_command(COMMAND) is not None
    script.write_text("print('tampered')\n")
    with execution_scope("job-a", "run-a", profile):
        result = json.loads(terminal._run_camera_runtime_command_if_allowed(COMMAND, cwd=str(profile), timeout=5))
    assert result["exit_code"] == -1


def test_missing_profile_values_never_fall_back_to_environment(runtime, monkeypatch):
    profile, _ = runtime
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "b" * 64)
    token = set_secret_scope({})
    try:
        with execution_scope("job-a", "run-a", profile), pytest.raises(PermissionError):
            build_camera_semantic_runtime_env()
    finally:
        reset_secret_scope(token)


def test_scheduler_to_terminal_to_private_python_flow(runtime, monkeypatch):
    import cron.scheduler as scheduler

    profile, _ = runtime
    (profile / ".env").write_text(
        "ZETTLAB_AGENT_ACTION_TOKEN=" + "a" * 64 + "\n"
        + "ZET_AGENT_OUTPUT_DIR=" + str(profile / "output") + "\n"
        + "ZET_CHAT_APPEND_URL=http://127.0.0.1:19090/api/v1/internal/chat/append\n"
    )
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: profile)
    monkeypatch.setattr(scheduler, "create_execution", lambda *a, **k: {"id": "run-a"})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *a: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *a: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *a: str(profile / "result"))
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    results = []

    def execute(job, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context

        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(
                copy_context().run, terminal.terminal_tool,
                command=COMMAND, task_id="semantic-scheduler-flow",
            ).result()
        results.append(json.loads(result))
        return True, result, "finished", None

    monkeypatch.setattr(scheduler, "run_job", execute)
    assert scheduler.run_one_job({"id": "job-a"}) is True
    assert len(results) == 1
    assert results[0]["exit_code"] == 0
    assert json.loads(results[0]["output"])["run_ok"] is True


@pytest.mark.parametrize("flags", [{"background": True}, {"pty": True}])
def test_semantic_helper_never_uses_background_shell(runtime, flags):
    result = json.loads(terminal.terminal_tool(command=COMMAND, task_id="semantic-no-shell", **flags))
    assert result["exit_code"] == -1
    assert result["camera_runtime_blocked"] is True


@pytest.mark.parametrize("callback", [
    "https://127.0.0.1:19090/callback", "http://192.168.1.1:19090/callback",
    "http://user:pass@127.0.0.1:19090/callback", "http://127.0.0.1:19090/callback?token=x",
    "http://127.0.0.1:invalid/callback", "",
])
def test_semantic_env_rejects_untrusted_callback(runtime, callback):
    from agent.secret_scope import current_secret_scope

    values = dict(current_secret_scope())
    values["ZET_CHAT_APPEND_URL"] = callback
    token = set_secret_scope(values)
    try:
        with execution_scope("job-a", "run-a", runtime[0]), pytest.raises(PermissionError):
            build_camera_semantic_runtime_env()
    finally:
        reset_secret_scope(token)


def test_generic_subprocess_scrubs_execution_identity(runtime):
    from tools.environments.local import _apply_profile_secret_scope_env

    env = {"ZETTLAB_CAMERA_JOB_ID": "forged", "ZETTLAB_CAMERA_EXECUTION_ID": "forged"}
    _apply_profile_secret_scope_env(env, inject=True)
    assert "ZETTLAB_CAMERA_JOB_ID" not in env
    assert "ZETTLAB_CAMERA_EXECUTION_ID" not in env
