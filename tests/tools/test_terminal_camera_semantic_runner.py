import json
import os
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
    from types import SimpleNamespace
    from agent.zet_agent_response_mode import trusted_skill_operation_block_message

    profile, _ = runtime
    with execution_scope("job-a", "run-a", profile):
        assert trusted_skill_operation_block_message(
            SimpleNamespace(), function_name="terminal", function_args={"command": COMMAND}
        ) is None
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
    assert terminal._parse_camera_runtime_command(prefix + " --matched false --unknown true") is not None
    proof = " --frame-states present,present,present,present --track-ids t,t,t,t"
    assert terminal._parse_camera_runtime_command(positive + proof) is not None
    assert terminal._parse_camera_runtime_command(positive + proof.replace("t,t,t,t", "t,t,/private,t")) is None
    assert terminal._parse_camera_runtime_command(positive + proof.replace("t,t,t,t", "t,t,t")) is None
    assert terminal._parse_camera_runtime_command(positive + proof + " --unknown true") is None


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "2.5", "2147483648", "01"])
def test_observation_parser_rejects_invalid_budget(runtime, budget):
    command = COMMAND.replace(" candidate ", " observe ") + " --timeout-seconds " + budget
    assert terminal._parse_camera_runtime_command(command) is None


@pytest.mark.parametrize("requested,limit,budget,expected", [
    (600, 600, 590, 595), (3600, 3600, 3500, 3505),
    (80, 600, 590, None), (600, 600, 600, None), (3600, 600, 3500, None),
])
def test_observation_budget_is_never_silently_clamped(requested, limit, budget, expected):
    from gateway.platforms.zet_agent_camera_semantic_arguments import semantic_execution_timeout
    args = ["observe", "--policy-id", "12345678-1234-1234-1234-123456789abc", "--timeout-seconds", str(budget)]
    if expected is None:
        with pytest.raises(ValueError):
            semantic_execution_timeout(args, requested, limit, 80)
    else:
        assert semantic_execution_timeout(args, requested, limit, 80) == expected
    assert semantic_execution_timeout(["candidate"], requested, limit, 80) == 80
    assert semantic_execution_timeout([], requested, limit, 80) == 80


def test_observation_rejects_insufficient_tool_timeout_before_child(runtime, monkeypatch):
    monkeypatch.setattr("tools.trusted_direct_runner.run_trusted_python_script", lambda **_: pytest.fail("must not spawn"))
    command = COMMAND.replace(" candidate ", " observe ") + " --timeout-seconds 590"
    with execution_scope("job-a", "run-a", runtime[0]):
        result = json.loads(terminal.terminal_tool(command=command, timeout=80, task_id="observe-short"))
    assert result["exit_code"] == -1


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


@pytest.mark.parametrize("scheduled", [False, True])
def test_real_middleware_preserves_semantic_runner_authorization(runtime, monkeypatch, scheduled):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from agent import tool_executor

    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *a, **k: None)
    agent = SimpleNamespace(
        platform="zet_agent", session_id="semantic-session",
        _tool_guardrails=SimpleNamespace(before_call=lambda *a: SimpleNamespace(allows_execution=True)),
    )
    context = execution_scope("job-a", "run-a", runtime[0]) if scheduled else nullcontext()
    with context:
        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent, function_name="terminal", function_args={"command": COMMAND},
            effective_task_id="semantic-middleware", tool_call_id="semantic-call",
            execute=lambda args: terminal.terminal_tool(**args, task_id="semantic-middleware"),
        )
    assert not outcome.blocked
    result = json.loads(outcome.result)
    assert result["camera_runtime_direct"] is True
    assert result["exit_code"] == (0 if scheduled else -1)


@pytest.mark.parametrize("matched", [False, True])
def test_real_presets_helper_candidate_and_commit(runtime, matched):
    """Opt-in cross-repo contract flow; never substitute an embedded helper.

    Run with HERMES_TEST_CAMERA_PRESETS_SOURCE pointing at the Presets
    script under test. The HTTP camera service and media are fixtures, while
    Python isolation, private FDs, script execution, HTTP and file IO are real.
    """
    import base64
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from agent.secret_scope import current_secret_scope

    source = os.environ.get("HERMES_TEST_CAMERA_PRESETS_SOURCE")
    if not source:
        pytest.skip("explicit Presets source is required for cross-repo flow")
    profile, script = runtime
    raw = Path(source).read_bytes()
    assert raw and len(raw) < 1024 * 1024
    script.write_bytes(raw)
    output = profile / "output"
    output.mkdir(mode=0o700)
    image = b"\xff\xd8fixture-media\xff\xd9"
    capability = "c" * 48
    times = [f"2026-09-07T12:00:{second:02d}Z" for second in (0, 4, 8, 12)]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("X-Zettlab-Agent-Action-Token")))
            if self.path.endswith("/semantic-candidates"):
                data = {
                    "capability": capability, "evidence_ref": "window-frame-0.jpg",
                    "frame_times": times, "subject_kind": "person", "subject_ref": "",
                    "predicate": "lingers", "zone_id": "", "min_duration_seconds": 10,
                    "image_data_uri": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
                }
            elif self.path.endswith("/semantic-observations"):
                data = {"analysis_complete": True, "analysis": {"unknown_batches": 1, "created_events": 0}}
            else:
                data = {"matched": body["verdict"]["matched"]}
            response = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    values = dict(current_secret_scope())
    values["ZET_CHAT_APPEND_URL"] = f"http://127.0.0.1:{server.server_port}/api/v1/internal/chat/append"
    token = set_secret_scope(values)
    try:
        with execution_scope("job-a", "run-a", profile):
            candidate = json.loads(terminal.terminal_tool(command=COMMAND, timeout=60, task_id="real-presets-candidate"))
            assert candidate["exit_code"] == 0, candidate
            data = json.loads(candidate["output"])["data"]
            attachment = Path(data["attachment_path"])
            assert attachment.parent == output
            assert attachment.read_bytes() == image
            assert attachment.stat().st_mode & 0o777 == 0o600
            assert data["frame_times"] == times
            assert "image_data_uri" not in data
            verdict = {"matched": matched}
            commit = f'python3 "$ZETTLAB_PRESETS_DIR/{SCRIPT}" commit --capability {data["capability"]} --matched {str(matched).lower()}'
            if matched:
                commit += " --subject-kind person --predicate lingers --duration-seconds 12 --evidence-ref " + data["evidence_ref"]
                commit += " --frame-states present,present,present,present --track-ids t,t,t,t"
                verdict.update(subject_kind="person", subject_ref="", predicate="lingers",
                               zone_id="", duration_seconds=12, evidence_ref=data["evidence_ref"])
                verdict["frames"] = [{"state": "present", "track_id": "t"} for _ in range(4)]
            result = json.loads(terminal.terminal_tool(command=commit, task_id="real-presets-commit"))
            assert result["exit_code"] == 0, result
            assert json.loads(result["output"]) == {"data": {"matched": matched}}
            observe = COMMAND.replace(" candidate ", " observe ") + " --timeout-seconds 590"
            observed = json.loads(terminal.terminal_tool(command=observe, timeout=600, task_id="real-presets-observe"))
            assert observed["exit_code"] == 0, observed
            assert json.loads(observed["output"]) == {"data": {
                "analysis_complete": True, "analysis": {"unknown_batches": 1, "created_events": 0},
            }}
        assert len(requests) == 3
        for path, body, bearer in requests:
            assert path.startswith("/api/v1/agent/hardware-connectors/cameras/semantic-")
            assert body["job_id"] == "job-a" and body["execution_id"] == "run-a"
            assert bearer == "a" * 64
        assert requests[1][1] == {
            "job_id": "job-a", "execution_id": "run-a", "capability": capability,
            "verdict": verdict,
        }
        assert requests[2][1] == {
            "policy_id": "12345678-1234-1234-1234-123456789abc",
            "execution_timeout_seconds": 590, "job_id": "job-a", "execution_id": "run-a",
        }
        assert terminal._CONNECTOR_RUNTIME_ROOT_ANCHOR.file_digests[SCRIPT] == hashlib.sha256(raw).hexdigest()
    finally:
        reset_secret_scope(token)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
