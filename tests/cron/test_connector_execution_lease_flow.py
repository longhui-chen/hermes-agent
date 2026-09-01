"""Regression coverage for non-Chat Connector Cron direct execution."""

import cron.connector_execution as connector_execution
import cron.scheduler as scheduler
import tools.cronjob_tools as cronjob_tools


def test_prepare_route_uses_only_ac_local_agent_identity(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    seen = {}

    def bridge_call(method, params, **kwargs):
        seen.update(method=method, params=params, kwargs=kwargs)
        return {"route_capability": "ac-local-route"}

    monkeypatch.setattr(connector_execution, "_bridge_call", bridge_call)

    capability = connector_execution.prepare_route_capability("job-1", "run-1", "github")

    assert capability == "ac-local-route"
    assert seen["method"] == "zettlab/cron/prepare-connector-execution"
    assert seen["params"] == {"job_id": "job-1", "execution_id": "run-1", "provider_id": "github"}
    assert "route_capability" not in seen["params"]


def test_connector_cron_prepares_direct_route_without_persisted_grant(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(connector_execution, "prepare_route_capability", lambda *_args: "ac-local-route")
    monkeypatch.setattr(connector_execution, "connector_provider_for_skills", lambda _skills: "github")

    capability, error = scheduler._prepare_connector_execution(
        {"id": "github-digest", "skills": ["GitHub Issue Read Actions"], "_connector_execution_id": "run-1"}
    )

    assert capability == "ac-local-route"
    assert error is None


def test_connector_cron_does_not_require_a_live_chat_grant():
    assert connector_execution.requires_live_chat_grant(["GitHub Issue Read Actions"]) is False


def test_resolves_provider_from_preset_manifest_not_skill_path(monkeypatch, tmp_path):
    manifest = tmp_path / "skills" / "custom-name" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "id: release-notes\nconnector_action_manifest:\n  provider_id: github\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path))

    assert connector_execution.connector_provider_for_skills(["release-notes", "ordinary-skill"]) == "github"


def test_mixed_connector_cron_stops_before_any_partial_execution(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(
        connector_execution,
        "connector_provider_for_skills",
        lambda _skills: (_ for _ in ()).throw(connector_execution.ConnectorExecutionLeaseError("task_connector_mixed_skills_unsupported")),
    )

    capability, error = scheduler._prepare_connector_execution(
        {"id": "mixed-connectors", "skills": ["Linear", "Gmail"]}
    )

    assert capability == ""
    assert error == "task_connector_mixed_skills_unsupported"


def test_create_does_not_persist_a_connector_grant(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(connector_execution, "connector_provider_for_skills", lambda _skills: "linear")
    saved = {}

    def create_job(**kwargs):
        saved.update(kwargs)
        return {"id": "job-1", **kwargs}

    monkeypatch.setattr(cronjob_tools, "create_job", create_job)
    monkeypatch.setattr(cronjob_tools, "_notify_provider_jobs_changed_safe", lambda: None)

    result = cronjob_tools.cronjob(action="create", schedule="every 1 hour", skills=["Linear"])

    assert '"success": true' in result
    assert "connector_execution" not in saved
