import agent.secret_scope as secret_scope
import hermes_cli.config as config_module
import tools.approval as approval


def test_multiplex_approval_state_uses_profile_and_session_key(monkeypatch):
    current_scope = ["/profiles/agent-a"]
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    monkeypatch.setattr(approval, "_approval_profile_scope", lambda: current_scope[0])
    for state in (
        approval._pending,
        approval._one_shot_approved,
        approval._session_approved,
        approval._gateway_queues,
        approval._gateway_notify_cbs,
        approval._permanent_approved_by_profile,
        approval._permanent_loaded_profiles,
    ):
        state.clear()

    session_key = "shared-session-key"
    approval.approve_session(session_key, "profile-a-only")
    approval.register_gateway_notify(session_key, lambda _data: None)
    approval_id = approval.submit_pending(
        session_key,
        {"pattern_key": "one-shot-a", "one_shot": True},
    )
    assert approval_id

    current_scope[0] = "/profiles/agent-b"
    assert not approval.is_approved(session_key, "profile-a-only")
    assert approval.approval_session_key_for_id(approval_id) is None
    assert approval.resolve_gateway_approval(
        session_key,
        "once",
        approval_id=approval_id,
    ) == 0
    callback_b = lambda _data: None
    approval.register_gateway_notify(session_key, callback_b)
    assert approval._gateway_notify_cbs[("/profiles/agent-b", session_key)] is callback_b

    current_scope[0] = "/profiles/agent-a"
    assert approval.is_approved(session_key, "profile-a-only")
    assert approval.approval_session_key_for_id(approval_id) == session_key
    assert approval.resolve_gateway_approval(
        session_key,
        "once",
        approval_id=approval_id,
    ) == 1
    assert approval._consume_one_shot_approval(session_key, "one-shot-a")
    for state in (
        approval._pending,
        approval._one_shot_approved,
        approval._session_approved,
        approval._gateway_queues,
        approval._gateway_notify_cbs,
        approval._permanent_approved_by_profile,
        approval._permanent_loaded_profiles,
    ):
        state.clear()


def test_multiplex_permanent_approvals_are_loaded_and_saved_per_profile(monkeypatch):
    current_scope = ["/profiles/agent-a"]
    configs = {
        "/profiles/agent-a": {"command_allowlist": ["profile-a-config"]},
        "/profiles/agent-b": {"command_allowlist": ["profile-b-config"]},
    }
    saved = {}
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    monkeypatch.setattr(approval, "_approval_profile_scope", lambda: current_scope[0])
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: dict(configs[current_scope[0]]),
    )
    monkeypatch.setattr(
        config_module,
        "save_config",
        lambda config: saved.__setitem__(current_scope[0], dict(config)),
    )
    approval._permanent_approved_by_profile.clear()
    approval._permanent_loaded_profiles.clear()

    assert approval.is_approved("shared", "profile-a-config")
    assert not approval.is_approved("shared", "profile-b-config")
    approval.approve_permanent("profile-a-added")
    approval.save_permanent_allowlist()

    current_scope[0] = "/profiles/agent-b"
    assert approval.is_approved("shared", "profile-b-config")
    assert not approval.is_approved("shared", "profile-a-config")
    assert not approval.is_approved("shared", "profile-a-added")
    approval.approve_permanent("profile-b-added")
    approval.save_permanent_allowlist()

    assert set(saved["/profiles/agent-a"]["command_allowlist"]) == {
        "profile-a-config",
        "profile-a-added",
    }
    assert set(saved["/profiles/agent-b"]["command_allowlist"]) == {
        "profile-b-config",
        "profile-b-added",
    }
    approval._permanent_approved_by_profile.clear()
    approval._permanent_loaded_profiles.clear()


def test_profile_unload_revokes_all_approval_state_before_same_path_reuse(monkeypatch):
    scope = ["/profiles/reused"]
    config = [{"command_allowlist": ["old-permanent"]}]
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    monkeypatch.setattr(approval, "_approval_profile_scope", lambda: scope[0])
    monkeypatch.setattr(config_module, "load_config", lambda: dict(config[0]))
    monkeypatch.setattr(config_module, "save_config", lambda _config: None)
    for state in (
        approval._pending,
        approval._one_shot_approved,
        approval._session_approved,
        approval._gateway_queues,
        approval._gateway_notify_cbs,
        approval._permanent_approved_by_profile,
        approval._permanent_loaded_profiles,
    ):
        state.clear()
    approval._session_yolo.clear()

    session_key = "same-session"
    assert approval.is_approved(session_key, "old-permanent")
    approval.approve_session(session_key, "old-session")
    approval.enable_session_yolo(session_key)
    approval._grant_one_shot_approval(session_key, "old-once")
    approval.register_gateway_notify(session_key, lambda _data: None)
    approval_id = approval.submit_pending(
        session_key, {"pattern_key": "old-pending", "one_shot": True}
    )
    state_key = (scope[0], session_key)
    live_entry = approval._ApprovalEntry({"approval_id": "live"})
    approval._gateway_queues[state_key] = [live_entry]

    removed = approval.purge_profile_approval_state(scope[0])
    assert removed["pending"] == 1
    assert removed["one_shot_grants"] == 1
    assert removed["session_allowlists"] == 1
    assert removed["session_yolo"] == 1
    assert removed["gateway_waiters"] == 1
    assert removed["gateway_callbacks"] == 1
    assert removed["permanent_allowlist"] == 1
    assert live_entry.event.is_set()
    assert live_entry.result == "deny"
    assert approval.approval_session_key_for_id(approval_id) is None

    config[0] = {"command_allowlist": ["new-permanent"]}
    assert approval.is_approved(session_key, "new-permanent")
    assert not approval.is_approved(session_key, "old-permanent")
    assert not approval.is_approved(session_key, "old-session")
    assert not approval.is_session_yolo_enabled(session_key)
