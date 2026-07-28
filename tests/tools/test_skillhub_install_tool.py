import json

import pytest

from tools import skillhub_install_tool as tool


@pytest.fixture(autouse=True)
def clear_pending_install_intents():
    with tool._pending_intents_lock:
        tool._pending_intents.clear()
    yield
    with tool._pending_intents_lock:
        tool._pending_intents.clear()


def test_rejects_short_name_and_direct_url(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid identity must not reach installer")
        ),
    )

    for identifier in (
        "ambiguous",
        "https://github.com/owner/repo",
        "official/agent/example",
    ):
        result = json.loads(tool.skillhub_install(identifier))
        assert "error" in result


def test_reports_install_from_authoritative_lock_delta(monkeypatch):
    identifier = "owner/repo/example"
    states = [
        {},
        {
            "example": {
                "identifier": identifier,
                "source": "github",
                "trust_level": "community",
                "scan_verdict": "safe",
                "content_hash": "sha256:abc",
                "install_path": "example",
            }
        },
    ]
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: states.pop(0))
    monkeypatch.setattr(tool, "_verify_entries_on_disk", lambda entries: entries)
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda exact, console=None, **kwargs: console.print(f"Installed {exact}"),
    )

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-1",
            user_message=f"install {identifier}",
        )
    )

    assert result["status"] == "installed"
    assert result["identifier"] == identifier
    assert result["skill_name"] == "example"
    assert result["content_hash"] == "sha256:abc"


def test_lock_read_failure_after_flow_is_unknown_and_not_retryable(monkeypatch):
    states = [{}, None]
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: states.pop(0))
    monkeypatch.setattr(tool, "_verify_entries_on_disk", lambda entries: entries)
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda exact, console=None, **kwargs: console.print(
            "approval and install flow returned"
        ),
    )

    result = json.loads(
        tool.skillhub_install(
            "owner/repo/example",
            session_id="session-1",
            turn_id="turn-1",
            user_message="install owner/repo/example",
        )
    )

    assert result["status"] == "unknown_reconcile"
    assert "Do not retry" in result["message"]


def test_existing_identical_entry_is_noop(monkeypatch):
    entry = {
        "example": {
            "identifier": "owner/repo/example",
            "content_hash": "sha256:abc",
        }
    }
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: entry)
    monkeypatch.setattr(tool, "_verify_entries_on_disk", lambda entries: entries)
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda exact, console=None: console.print("already installed"),
    )

    result = json.loads(
        tool.skillhub_install(
            "owner/repo/example",
            session_id="session-1",
            turn_id="turn-1",
            user_message="install owner/repo/example",
        )
    )

    assert result["status"] == "already_installed"


def test_existing_lock_with_unverified_disk_state_is_unknown(monkeypatch):
    entry = {
        "example": {
            "identifier": "owner/repo/example",
            "content_hash": "sha256:stale",
            "install_path": "example",
        }
    }
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: entry)
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unverified state must not reach installer")
        ),
    )

    result = json.loads(
        tool.skillhub_install(
            "owner/repo/example",
            session_id="session-1",
            turn_id="turn-1",
            user_message="install owner/repo/example",
        )
    )

    assert result["status"] == "unknown_reconcile"
    assert "no mutation was attempted" in result["message"]


def test_installer_exception_is_unknown_and_not_retryable(monkeypatch):
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("lock write failed")),
    )

    result = json.loads(
        tool.skillhub_install(
            "owner/repo/example",
            session_id="session-1",
            turn_id="turn-1",
            user_message="install owner/repo/example",
        )
    )

    assert result["status"] == "unknown_reconcile"
    assert "Do not retry" in result["message"]
    assert result["installer_error"] == "OSError: lock write failed"


def test_unconfirmed_call_prepares_candidate_for_later_user_turn(monkeypatch):
    identifier = "owner/repo/example"
    candidate = {
        "identifier": identifier,
        "source_url": "https://github.com/owner/repo/tree/main/example",
        "content_hash": "sha256:prepared",
        "scan_verdict": "safe",
    }
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda exact, console=None, **kwargs: {
            "status": "confirmation_required",
            "candidate": candidate,
        },
    )

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-1",
            user_message="帮我找一个 HTML skill",
        )
    )

    assert result["status"] == "confirmation_required"
    assert result["candidate"] == candidate
    pending = tool._get_pending_intent("session-1", identifier)
    assert pending is not None
    assert pending.created_turn_id == "turn-1"


def test_later_affirmative_user_turn_consumes_exact_pending_candidate(monkeypatch):
    identifier = "owner/repo/example"
    candidate = {
        "identifier": identifier,
        "source_url": "https://github.com/owner/repo/tree/main/example",
        "content_hash": "sha256:prepared",
        "scan_verdict": "safe",
    }
    tool._store_pending_intent(
        session_id="session-1",
        turn_id="turn-1",
        identifier=identifier,
        candidate=candidate,
    )
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    observed = {}

    def install(exact, console=None, **kwargs):
        observed.update(kwargs)
        return {"status": "risk_denied", "candidate": candidate}

    monkeypatch.setattr("hermes_cli.skills_hub.do_agent_install", install)

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-2",
            user_message="确认安装",
            previous_assistant_message=(
                f"候选是 {identifier}，来源 "
                "https://github.com/owner/repo/tree/main/example，是否安装？"
            ),
        )
    )

    assert result["status"] == "risk_denied"
    assert observed["intent_confirmed"] is True
    assert observed["expected_candidate"] == candidate
    assert tool._get_pending_intent("session-1", identifier) is None


def test_same_turn_or_undisclosed_candidate_cannot_consume_pending(monkeypatch):
    identifier = "owner/repo/example"
    candidate = {
        "identifier": identifier,
        "source_url": "https://github.com/owner/repo/tree/main/example",
        "content_hash": "sha256:prepared",
        "scan_verdict": "safe",
    }
    tool._store_pending_intent(
        session_id="session-1",
        turn_id="turn-1",
        identifier=identifier,
        candidate=candidate,
    )
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    observed = []

    def prepare(exact, console=None, **kwargs):
        observed.append(kwargs["intent_confirmed"])
        return {"status": "confirmation_required", "candidate": candidate}

    monkeypatch.setattr("hermes_cli.skills_hub.do_agent_install", prepare)

    same_turn = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-1",
            user_message="确认安装",
            previous_assistant_message=(
                f"候选是 {identifier}，来源 "
                "https://github.com/owner/repo/tree/main/example"
            ),
        )
    )
    later_but_undisclosed = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-2",
            user_message="确认安装",
            previous_assistant_message=(
                f"候选是 {identifier}，但这里只披露了另一个来源 "
                "https://github.com/owner/repo/tree/main/other"
            ),
        )
    )

    assert same_turn["status"] == "confirmation_required"
    assert later_but_undisclosed["status"] == "confirmation_required"
    assert observed == [False, False]


def test_explicit_user_message_with_exact_identifier_is_direct_intent(monkeypatch):
    identifier = "owner/repo/example"
    candidate = {"identifier": identifier, "scan_verdict": "safe"}
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    observed = {}

    def install(exact, console=None, **kwargs):
        observed.update(kwargs)
        return {"status": "risk_denied", "candidate": candidate}

    monkeypatch.setattr("hermes_cli.skills_hub.do_agent_install", install)

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-1",
            user_message=f"请安装 {identifier}",
        )
    )

    assert result["status"] == "risk_denied"
    assert observed["intent_confirmed"] is True
    assert observed["expected_candidate"] is None


@pytest.mark.parametrize(
    "message",
    [
        "Should I install owner/repo/example?",
        "Why install owner/repo/example?",
        "What happens if I install owner/repo/example?",
        "owner/repo/example 安装安全吗？",
        "请介绍如何安装 owner/repo/example",
    ],
)
def test_question_or_explanation_is_not_direct_install_intent(
    monkeypatch, message
):
    identifier = "owner/repo/example"
    candidate = {
        "identifier": identifier,
        "source_url": "https://github.com/owner/repo/tree/main/example",
        "content_hash": "sha256:prepared",
        "scan_verdict": "safe",
    }
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    observed = {}

    def prepare(exact, console=None, **kwargs):
        observed.update(kwargs)
        return {"status": "confirmation_required", "candidate": candidate}

    monkeypatch.setattr("hermes_cli.skills_hub.do_agent_install", prepare)

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-1",
            user_message=message,
        )
    )

    assert result["status"] == "confirmation_required"
    assert observed["intent_confirmed"] is False


def test_negative_user_turn_cancels_pending_without_installer(monkeypatch):
    identifier = "owner/repo/example"
    tool._store_pending_intent(
        session_id="session-1",
        turn_id="turn-1",
        identifier=identifier,
        candidate={"identifier": identifier},
    )
    monkeypatch.setattr(tool, "_read_lock_entries", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.skills_hub.do_agent_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("declined intent must not reach installer")
        ),
    )

    result = json.loads(
        tool.skillhub_install(
            identifier,
            session_id="session-1",
            turn_id="turn-2",
            user_message="不要安装了",
            previous_assistant_message=f"候选是 {identifier}",
        )
    )

    assert result["status"] == "confirmation_declined"
    assert tool._get_pending_intent("session-1", identifier) is None


def test_install_mutation_primitive_rejects_unauthorized_direct_call(
    monkeypatch, tmp_path
):
    import tools.skills_hub as hub
    from tools.skills_guard import ScanResult

    skills_dir = tmp_path / "skills"
    quarantine_root = skills_dir / ".hub" / "quarantine"
    quarantine_path = quarantine_root / "example"
    quarantine_path.mkdir(parents=True)
    (quarantine_path / "SKILL.md").write_text("# Example\n")
    existing_install = skills_dir / "example"
    existing_install.mkdir(parents=True)
    (existing_install / "keep.txt").write_text("existing\n")
    monkeypatch.setattr(hub, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub, "QUARANTINE_DIR", quarantine_root)

    bundle = hub.SkillBundle(
        name="example",
        files={"SKILL.md": "# Example\n"},
        source="github",
        identifier="owner/repo/example",
        trust_level="community",
    )
    scan = ScanResult(
        skill_name="example",
        source="github",
        trust_level="community",
        verdict="safe",
    )

    with pytest.raises(PermissionError, match="requires authorization"):
        hub.install_from_quarantine(
            quarantine_path,
            "example",
            "",
            bundle,
            scan,
        )
    assert quarantine_path.is_dir()
    assert (existing_install / "keep.txt").read_text() == "existing\n"


def test_install_mutation_primitive_rejects_authorization_for_other_source(
    monkeypatch, tmp_path
):
    import tools.skills_hub as hub
    from tools.skills_guard import ScanResult

    skills_dir = tmp_path / "skills"
    quarantine_root = skills_dir / ".hub" / "quarantine"
    quarantine_path = quarantine_root / "example"
    quarantine_path.mkdir(parents=True)
    (quarantine_path / "SKILL.md").write_text("# Example\n")
    monkeypatch.setattr(hub, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub, "QUARANTINE_DIR", quarantine_root)

    approved_bundle = hub.SkillBundle(
        name="example",
        files={"SKILL.md": "# Example\n"},
        source="github",
        identifier="owner/repo/example",
        trust_level="community",
        metadata={"source_url": "https://github.com/owner/repo/tree/approved/example"},
    )
    changed_source_bundle = hub.SkillBundle(
        name="example",
        files={"SKILL.md": "# Example\n"},
        source="github",
        identifier="owner/repo/example",
        trust_level="community",
        metadata={"source_url": "https://github.com/owner/repo/tree/changed/example"},
    )
    authorization = hub._issue_install_mutation_authorization(
        "example",
        approved_bundle,
        approved_source_url=hub.source_url_for_bundle(approved_bundle),
        approved_quarantine_hash=hub.full_content_hash(quarantine_path),
    )
    scan = ScanResult(
        skill_name="example",
        source="github",
        trust_level="community",
        verdict="safe",
    )

    with pytest.raises(PermissionError, match="requires authorization"):
        hub.install_from_quarantine(
            quarantine_path,
            "example",
            "",
            changed_source_bundle,
            scan,
            _authorization=authorization,
        )
    assert quarantine_path.is_dir()
    assert not (skills_dir / "example").exists()


def test_install_mutation_primitive_rejects_content_changed_after_approval(
    monkeypatch, tmp_path
):
    import tools.skills_hub as hub
    from tools.skills_guard import ScanResult

    skills_dir = tmp_path / "skills"
    quarantine_root = skills_dir / ".hub" / "quarantine"
    quarantine_path = quarantine_root / "example"
    quarantine_path.mkdir(parents=True)
    skill_md = quarantine_path / "SKILL.md"
    skill_md.write_text("# Approved\n")
    monkeypatch.setattr(hub, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub, "QUARANTINE_DIR", quarantine_root)

    bundle = hub.SkillBundle(
        name="example",
        files={"SKILL.md": "# Approved\n"},
        source="github",
        identifier="owner/repo/example",
        trust_level="community",
        metadata={"source_url": "https://github.com/owner/repo/tree/main/example"},
    )
    authorization = hub._issue_install_mutation_authorization(
        "example",
        bundle,
        approved_source_url=hub.source_url_for_bundle(bundle),
        approved_quarantine_hash=hub.full_content_hash(quarantine_path),
    )
    skill_md.write_text("# Changed after approval\n")
    scan = ScanResult(
        skill_name="example",
        source="github",
        trust_level="community",
        verdict="safe",
    )

    with pytest.raises(PermissionError, match="requires authorization"):
        hub.install_from_quarantine(
            quarantine_path,
            "example",
            "",
            bundle,
            scan,
            _authorization=authorization,
        )
    assert quarantine_path.is_dir()
    assert not (skills_dir / "example").exists()


def test_install_mutation_rolls_back_filesystem_when_lock_write_fails(
    monkeypatch, tmp_path
):
    import tools.skills_hub as hub
    from tools.skills_guard import ScanResult

    skills_dir = tmp_path / "skills"
    quarantine_root = skills_dir / ".hub" / "quarantine"
    quarantine_path = quarantine_root / "example"
    quarantine_path.mkdir(parents=True)
    (quarantine_path / "SKILL.md").write_text("# New candidate\n")
    existing_install = skills_dir / "example"
    existing_install.mkdir(parents=True)
    (existing_install / "SKILL.md").write_text("# Existing install\n")
    monkeypatch.setattr(hub, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub, "QUARANTINE_DIR", quarantine_root)
    monkeypatch.setattr(
        hub.HubLockFile,
        "record_install",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    bundle = hub.SkillBundle(
        name="example",
        files={"SKILL.md": "# New candidate\n"},
        source="github",
        identifier="owner/repo/example",
        trust_level="community",
    )
    authorization = hub._issue_install_mutation_authorization(
        "example",
        bundle,
        approved_source_url=hub.source_url_for_bundle(bundle),
        approved_quarantine_hash=hub.full_content_hash(quarantine_path),
    )
    scan = ScanResult(
        skill_name="example",
        source="github",
        trust_level="community",
        verdict="safe",
    )

    with pytest.raises(OSError, match="disk full"):
        hub.install_from_quarantine(
            quarantine_path,
            "example",
            "",
            bundle,
            scan,
            _authorization=authorization,
        )

    assert (existing_install / "SKILL.md").read_text() == "# Existing install\n"
    assert (quarantine_path / "SKILL.md").read_text() == "# New candidate\n"
