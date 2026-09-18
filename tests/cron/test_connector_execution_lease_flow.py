"""Regression coverage for non-Chat Connector Cron direct execution."""

import pytest

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


def test_mixed_connector_cron_uses_agent_level_route_instead_of_blocking(monkeypatch, tmp_path):
    for skill_id, provider in (("linear-notes", "linear"), ("mail-digest", "gmail")):
        manifest = tmp_path / "skills" / skill_id / "manifest.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            f"id: {skill_id}\nconnector_action_manifest:\n  provider_id: {provider}\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path))
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    seen = {}

    def prepare(job_id, execution_id, provider_id):
        seen.update(job_id=job_id, execution_id=execution_id, provider_id=provider_id)
        return "ac-local-agent-route"

    monkeypatch.setattr(connector_execution, "prepare_route_capability", prepare)

    assert connector_execution.connector_provider_for_skills(["linear-notes", "mail-digest"]) == ""

    capability, error = scheduler._prepare_connector_execution(
        {"id": "mixed-connectors", "skills": ["linear-notes", "mail-digest"], "_connector_execution_id": "run-3"}
    )

    assert (capability, error) == ("ac-local-agent-route", None)
    assert seen == {"job_id": "mixed-connectors", "execution_id": "run-3", "provider_id": ""}


def test_create_does_not_persist_a_connector_grant(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(connector_execution, "connector_provider_for_skills", lambda _skills: "linear")
    saved = {}

    def create_job(**kwargs):
        saved.update(kwargs)
        job = dict(kwargs)
        job.update(
            id="job-1",
            name=kwargs.get("name") or "Linear",
            schedule_display="every 1 hour",
            next_run_at="2026-09-02T00:00:00Z",
            repeat={"times": kwargs.get("repeat"), "completed": 0},
        )
        return job

    monkeypatch.setattr(cronjob_tools, "create_job", create_job)
    monkeypatch.setattr(cronjob_tools, "_notify_provider_jobs_changed_safe", lambda: None)

    result = cronjob_tools.cronjob(action="create", schedule="every 1 hour", skills=["Linear"])

    assert '"success": true' in result
    assert "connector_execution" not in saved


def test_empty_skills_prepare_agent_level_route(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", "")
    seen = {}

    def prepare(job_id, execution_id, provider_id):
        seen.update(job_id=job_id, execution_id=execution_id, provider_id=provider_id)
        return "ac-local-agent-route"

    monkeypatch.setattr(connector_execution, "prepare_route_capability", prepare)

    assert connector_execution.connector_provider_for_skills([]) == ""

    capability, error = scheduler._prepare_connector_execution(
        {"id": "mail-sync", "skills": [], "_connector_execution_id": "run-7"}
    )

    assert (capability, error) == ("ac-local-agent-route", None)
    assert seen == {"job_id": "mail-sync", "execution_id": "run-7", "provider_id": ""}


def test_prepare_route_sends_empty_provider_for_agent_level_route(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    seen = {}

    def bridge_call(method, params, **kwargs):
        seen.update(method=method, params=params)
        return {"route_capability": "ac-local-agent-route"}

    monkeypatch.setattr(connector_execution, "_bridge_call", bridge_call)

    assert connector_execution.prepare_route_capability("job-1", "run-1", "") == "ac-local-agent-route"
    assert seen["method"] == "zettlab/cron/prepare-connector-execution"
    assert seen["params"] == {"job_id": "job-1", "execution_id": "run-1", "provider_id": ""}


def test_prepare_route_still_rejects_malformed_provider(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)

    def bridge_call(*_args, **_kwargs):
        raise AssertionError("broker must not be called for a malformed provider")

    monkeypatch.setattr(connector_execution, "_bridge_call", bridge_call)

    with pytest.raises(connector_execution.ConnectorExecutionLeaseError):
        connector_execution.prepare_route_capability("job-1", "run-1", "Bad Provider!")


def test_agent_level_route_failure_does_not_block_the_run(monkeypatch, caplog):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(connector_execution, "connector_provider_for_skills", lambda _skills: "")

    def prepare(*_args):
        raise connector_execution.ConnectorExecutionLeaseError("task_connector_not_authorized")

    monkeypatch.setattr(connector_execution, "prepare_route_capability", prepare)

    with caplog.at_level("WARNING", logger="cron.scheduler"):
        capability, error = scheduler._prepare_connector_execution(
            {"id": "daily-report", "skills": ["Ordinary Skill"]}
        )

    assert (capability, error) == ("", None)
    assert "agent-level connector route unavailable" in caplog.text


def test_single_provider_route_failure_still_blocks(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(connector_execution, "connector_provider_for_skills", lambda _skills: "github")

    def prepare(*_args):
        raise connector_execution.ConnectorExecutionLeaseError("task_connector_temporarily_unavailable", retryable=True)

    monkeypatch.setattr(connector_execution, "prepare_route_capability", prepare)

    capability, error = scheduler._prepare_connector_execution(
        {"id": "github-digest", "skills": ["GitHub Issue Read Actions"]}
    )

    assert capability == ""
    assert error == "task_connector_temporarily_unavailable"
