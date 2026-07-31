import threading
from pathlib import Path

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
    registry.on_close = lambda _session, _session_id: None
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
