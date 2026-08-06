import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import tools.process_registry as process_module
from tools.process_registry import ProcessRegistry, ProcessSession


def _registry() -> ProcessRegistry:
    registry = object.__new__(ProcessRegistry)
    registry._running = {}
    registry._finished = {}
    registry._lock = threading.Lock()
    registry._completion_consumed = set()
    registry._poll_observed = set()
    registry.pending_watchers = []
    registry.completion_queue = queue.Queue()
    registry.on_close = lambda _session, _session_id: None
    registry._write_checkpoint = lambda: None
    return registry


def test_process_actions_and_list_are_profile_owned(monkeypatch, tmp_path):
    registry = _registry()
    main = (tmp_path / "main").resolve()
    coder = (tmp_path / "coder").resolve()
    main.mkdir()
    coder.mkdir()
    active = [main]
    monkeypatch.setattr(process_module, "get_hermes_home", lambda: active[0])
    main_session = ProcessSession(
        id="proc_main",
        command="main",
        task_id="shared-task",
        session_key="shared-session",
        profile_owner=str(main),
        started_at=1.0,
    )
    coder_session = ProcessSession(
        id="proc_coder",
        command="coder",
        task_id="shared-task",
        session_key="shared-session",
        profile_owner=str(coder),
        started_at=1.0,
    )
    missing_owner = ProcessSession(
        id="proc_missing",
        command="missing",
        profile_owner="",
        started_at=1.0,
    )
    registry._running = {
        main_session.id: main_session,
        coder_session.id: coder_session,
        missing_owner.id: missing_owner,
    }

    assert registry.get("proc_main") is main_session
    assert registry.get("proc_coder") is None
    assert registry.get("proc_missing") is None
    assert [item["session_id"] for item in registry.list_sessions()] == ["proc_main"]
    assert registry.poll("proc_coder")["status"] == "not_found"
    assert registry.read_log("proc_coder")["status"] == "not_found"
    assert registry.kill_process("proc_coder")["status"] == "not_found"
    assert registry.write_stdin("proc_coder", "x")["status"] == "not_found"
    assert registry.submit_stdin("proc_coder", "x")["status"] == "not_found"
    assert registry.close_stdin("proc_coder")["status"] == "not_found"

    active[0] = coder
    assert registry.get("proc_coder") is coder_session
    assert registry.get("proc_main") is None
    assert [item["session_id"] for item in registry.list_sessions()] == ["proc_coder"]


def test_recovery_without_profile_owner_fails_closed(monkeypatch, tmp_path):
    registry = _registry()
    checkpoint = tmp_path / "processes.json"
    checkpoint.write_text(
        '[{"session_id":"proc_legacy","command":"sleep","pid":123}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: True)

    assert registry.recover_from_checkpoint() == 0
    assert registry._running == {}


def test_managed_checkpoint_uses_broker_runtime_path(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    assert process_module._checkpoint_path() == Path(
        "/run/zettlab-claw/processes.json"
    )
    assert process_module._checkpoint_path() != process_module.CHECKPOINT_PATH


def test_managed_recovery_rejects_wrong_uid_or_cgroup(monkeypatch, tmp_path):
    registry = _registry()
    checkpoint = tmp_path / "processes.json"
    owner = str((tmp_path / "profiles" / "coder").resolve())
    checkpoint.write_text(
        __import__("json").dumps([{
            "session_id": "proc_forged",
            "command": "sleep",
            "pid": 123,
            "pid_scope": "host",
            "host_start_time": 456,
            "profile_owner": owner,
        }]),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(process_module, "_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(
        process_module, "_validate_managed_checkpoint_path", lambda _path: None
    )
    monkeypatch.setattr(
        process_module, "_managed_profile_owner_allowed", lambda _owner: True
    )
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: True)
    monkeypatch.setattr(
        registry, "_managed_pid_matches_profile", lambda *_args: False
    )

    assert registry.recover_from_checkpoint() == 0
    assert registry._running == {}


def test_managed_pid_matches_service_uid_and_profile_cgroup(monkeypatch, tmp_path):
    from tools.environments import local as local_module

    pid = 123
    proc_root = tmp_path / "proc"
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True)
    (process_root / "status").write_text(
        "State:\tS (sleeping)\nUid:\t0\t0\t0\t0\n",
        encoding="utf-8",
    )
    (process_root / "cgroup").write_text(
        "0::/system.slice/zettlab-claw.service/terminal-profile-100001\n",
        encoding="ascii",
    )
    monkeypatch.setattr(process_module, "_PROC_ROOT", proc_root)
    monkeypatch.setattr(process_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_identity",
        lambda _env=None: (100001, 100001),
    )
    monkeypatch.setenv(
        "HERMES_MANAGED_CGROUP_ROOT",
        "/system.slice/zettlab-claw.service",
    )

    assert ProcessRegistry._managed_pid_matches_profile(
        pid, "/profiles/coder"
    )

    (process_root / "status").write_text(
        "State:\tS (sleeping)\nUid:\t100001\t100001\t100001\t100001\n",
        encoding="utf-8",
    )
    assert not ProcessRegistry._managed_pid_matches_profile(
        pid, "/profiles/coder"
    )


def test_global_cleanup_uses_only_preselected_registry_objects(monkeypatch, tmp_path):
    registry = _registry()
    main = (tmp_path / "main").resolve()
    coder = (tmp_path / "coder").resolve()
    main.mkdir()
    coder.mkdir()
    monkeypatch.setattr(process_module, "get_hermes_home", lambda: main)
    sessions = [
        ProcessSession(
            id="proc_main",
            command="main",
            profile_owner=str(main),
            started_at=1.0,
        ),
        ProcessSession(
            id="proc_coder",
            command="coder",
            profile_owner=str(coder),
            started_at=1.0,
        ),
    ]
    registry._running = {session.id: session for session in sessions}
    selected = []

    def fake_kill(session_id, **kwargs):
        selected.append((session_id, kwargs.get("_trusted_session")))
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", fake_kill)
    assert registry.kill_all(all_profiles=True) == 2
    assert selected == [(session.id, session) for session in sessions]


def test_explicit_profile_activity_and_purge_are_owner_scoped(tmp_path):
    registry = _registry()
    main = (tmp_path / "main").resolve()
    coder = (tmp_path / "coder").resolve()
    main.mkdir()
    coder.mkdir()
    main_session = ProcessSession(
        id="proc_main", command="main", task_id="shared-task",
        profile_owner=str(main), started_at=1.0,
    )
    coder_session = ProcessSession(
        id="proc_coder", command="coder", task_id="shared-task",
        profile_owner=str(coder), started_at=1.0,
    )
    coder_session.exited = True
    registry._running = {main_session.id: main_session}
    registry._finished = {coder_session.id: coder_session}
    registry._completion_consumed = {main_session.id, coder_session.id}
    registry._poll_observed = {main_session.id, coder_session.id}
    registry.pending_watchers = [
        {"session_id": main_session.id, "profile_owner": str(main)},
        {"session_id": coder_session.id, "profile_owner": str(coder)},
    ]
    registry.completion_queue.put(
        {"session_id": main_session.id, "profile_owner": str(main)}
    )
    registry.completion_queue.put(
        {"session_id": coder_session.id, "profile_owner": str(coder)}
    )

    assert registry.has_active_processes_for_profile("shared-task", str(main))
    assert not registry.has_active_processes_for_profile("shared-task", str(coder))
    removed = registry.purge_profile_state(str(coder))

    assert removed == {
        "running_records": 0,
        "finished_records": 1,
        "pending_watchers": 1,
        "completion_events": 1,
    }
    assert list(registry._running) == [main_session.id]
    assert registry._finished == {}
    assert registry._completion_consumed == {main_session.id}
    assert registry._poll_observed == {main_session.id}
    assert registry.pending_watchers == [
        {"session_id": main_session.id, "profile_owner": str(main)}
    ]
    assert registry.completion_queue.get_nowait()["session_id"] == main_session.id


def test_idle_reaper_uses_recorded_profile_owner(monkeypatch, tmp_path):
    import tools.terminal_tool as terminal_tool

    owner = str((tmp_path / "coder").resolve())
    key = "profile-key:default"
    cleaned = []
    environment = SimpleNamespace(cleanup=lambda: cleaned.append(True))
    monkeypatch.setattr(terminal_tool, "_active_environments", {key: environment})
    monkeypatch.setattr(terminal_tool, "_last_activity", {key: 0.0})
    monkeypatch.setattr(
        terminal_tool, "_environment_profile_owners", {key: owner}
    )
    monkeypatch.setattr(terminal_tool.time, "time", lambda: 100.0)
    calls = []

    def has_active(task_id, profile_owner):
        calls.append((task_id, profile_owner))
        return True

    monkeypatch.setattr(
        process_module.process_registry,
        "has_active_processes_for_profile",
        has_active,
    )
    terminal_tool._cleanup_inactive_envs(lifetime_seconds=10)

    assert calls == [(key, owner)]
    assert terminal_tool._last_activity[key] == 100.0
    assert cleaned == []
