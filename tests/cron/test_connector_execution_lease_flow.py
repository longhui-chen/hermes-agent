"""Regression coverage for task-scoped Linear Connector execution leases."""

import json

import cron.connector_execution as connector_execution
import cron.scheduler as scheduler
import tools.cronjob_tools as cronjob_tools


def test_create_grant_keeps_only_task_scoped_material(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    seen = {}

    def bridge_call(method, params, **kwargs):
        seen.update(method=method, params=params, kwargs=kwargs)
        return {
            "grant_id": "grant-1",
            "grant_token": "opaque-task-grant",
            "expires_at": "2026-08-04T00:00:00Z",
        }

    monkeypatch.setattr(connector_execution, "_bridge_call", bridge_call)
    monkeypatch.setattr(
        "gateway.session_context.zettlab_connector_route_capability",
        lambda: "live-chat-route-capability",
    )

    grant = connector_execution.create_grant("job-1")

    assert seen["method"] == "zettlab/cron/create-connector-grant"
    assert seen["params"] == {"job_id": "job-1", "provider_id": "linear"}
    assert seen["kwargs"]["route_capability"] == "live-chat-route-capability"
    assert grant["grant_token"] == "opaque-task-grant"
    assert "bearer" not in grant


def test_legacy_linear_job_keeps_its_existing_execution_path_when_flag_is_enabled(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    job = {"id": "legacy-linear", "name": "Linear digest", "skills": ["Linear"]}

    capability, error = scheduler._prepare_connector_execution(job)

    assert capability == ""
    assert error is None


def test_legacy_singular_linear_skill_keeps_its_existing_execution_path(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)

    capability, error = scheduler._prepare_connector_execution(
        {"id": "legacy-singular", "name": "Linear digest", "skill": "Linear"}
    )

    assert capability == ""
    assert error is None


def test_migrated_linear_job_keeps_lease_path_when_flag_is_disabled(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: False)
    monkeypatch.setattr(connector_execution, "acquire_route_capability", lambda *_args: "task-route-capability")

    capability, error = scheduler._prepare_connector_execution(
        {
            "id": "migrated-linear",
            "skills": ["Linear"],
            "connector_execution": {"provider_id": "linear", "grant_id": "grant-1", "grant_token": "opaque"},
        }
    )

    assert capability == "task-route-capability"
    assert error is None


def test_shared_execution_path_restores_grant_without_leaking_it_to_public_job(monkeypatch):
    public_job = {
        "id": "migrated-linear",
        "skills": ["Linear"],
        "connector_execution": {"provider_id": "linear", "grant_id": "grant-1", "authorization_state": "authorized"},
    }
    persisted_job = {
        **public_job,
        "connector_execution": {"provider_id": "linear", "grant_id": "grant-1", "grant_token": "opaque"},
    }
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda job_id: persisted_job if job_id == "migrated-linear" else None)

    execution_job = scheduler._attach_private_connector_execution(public_job)

    assert execution_job["connector_execution"]["grant_token"] == "opaque"
    assert "grant_token" not in public_job["connector_execution"]


def test_migrated_mixed_connector_job_stops_before_any_partial_execution(monkeypatch):
    monkeypatch.setattr(connector_execution, "acquire_route_capability", lambda *_args: "must-not-run")

    capability, error = scheduler._prepare_connector_execution(
        {
            "id": "mixed-connectors",
            "skills": ["Linear", "Gmail"],
            "connector_execution": {"provider_id": "linear", "grant_id": "grant-1", "grant_token": "opaque"},
        }
    )

    assert capability == ""
    assert error == "task_connector_mixed_skills_unsupported"


def test_create_rejects_mixed_linear_connector_migration_before_persisting(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(
        cronjob_tools,
        "create_job",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("mixed job must not be persisted")),
    )

    response = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1 hour",
            skills=["Linear", "Gmail"],
        )
    )

    assert response["success"] is False
    assert "同时使用 Linear" in response["error"]


def test_update_rejects_mixed_linear_connector_migration_before_persisting(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    existing = {
        "id": "legacy-linear",
        "name": "old",
        "skills": ["Linear"],
        "skill": "Linear",
        "schedule": {"kind": "interval"},
    }
    monkeypatch.setattr(cronjob_tools, "resolve_job_ref", lambda _ref: dict(existing))
    monkeypatch.setattr(
        cronjob_tools,
        "update_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mixed update must not persist")),
    )

    response = json.loads(
        cronjob_tools.cronjob(
            action="update",
            job_id="legacy-linear",
            skills=["Linear", "Gmail"],
        )
    )

    assert response["success"] is False
    assert "同时使用 Linear" in response["error"]


def test_scheduler_keeps_network_failure_distinct_from_authorization(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setattr(
        connector_execution,
        "acquire_route_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            connector_execution.ConnectorExecutionLeaseError(
                "task_connector_temporarily_unavailable", retryable=True
            )
        ),
    )

    success, document, _final, error = scheduler.run_job(
        {
            "id": "linear-unavailable",
            "skills": ["Linear"],
            "connector_execution": {"provider_id": "linear", "grant_id": "grant-1", "grant_token": "opaque"},
        }
    )

    assert success is False
    assert error == "task_connector_temporarily_unavailable"
    assert "暂时不可用" in document
    assert "未读取数据、未生成报告" in document


def test_scheduler_stops_before_agent_when_lease_is_rejected(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)

    def reject(*_args, **_kwargs):
        raise connector_execution.ConnectorExecutionLeaseError(
            "task_connector_authorization_expired"
        )

    monkeypatch.setattr(connector_execution, "acquire_route_capability", reject)
    job = {
        "id": "linear-digest",
        "name": "Linear digest",
        "skills": ["Linear"],
        "connector_execution": {
            "provider_id": "linear",
            "grant_id": "grant-1",
            "grant_token": "opaque-task-grant",
        },
    }

    success, document, _final, error = scheduler.run_job(job)

    assert success is False
    assert error == "task_connector_authorization_expired"
    assert "未读取数据、未生成报告" in document


def test_update_migrates_existing_linear_job_under_live_route(monkeypatch):
    monkeypatch.setattr(connector_execution, "enabled", lambda: True)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    existing = {
        "id": "legacy-linear",
        "name": "old",
        "skills": ["Linear"],
        "skill": "Linear",
        "schedule": {"kind": "interval"},
    }
    saved = {}

    monkeypatch.setattr(cronjob_tools, "resolve_job_ref", lambda _ref: dict(existing))
    monkeypatch.setattr(cronjob_tools, "_notify_provider_jobs_changed_safe", lambda: None)
    monkeypatch.setattr(
        connector_execution,
        "create_grant",
        lambda job_id, provider: {
            "provider_id": provider,
            "grant_id": "grant-1",
            "grant_token": "opaque-task-grant",
            "expires_at": "2026-08-04T00:00:00Z",
        },
    )

    def save(job_id, updates):
        saved["job_id"] = job_id
        saved["updates"] = updates
        return {**existing, **updates}

    monkeypatch.setattr(cronjob_tools, "update_job", save)
    response = json.loads(cronjob_tools.cronjob(action="update", job_id="legacy-linear", name="new"))

    assert response["success"] is True
    assert saved["job_id"] == "legacy-linear"
    assert saved["updates"]["connector_execution"]["grant_token"] == "opaque-task-grant"
    assert "connector_execution" not in response["job"], "tool response must not leak the grant"


def test_read_job_shape_redacts_task_grant_token():
    from cron.jobs import _normalize_job_record

    normalized = _normalize_job_record(
        {
            "id": "linear-digest",
            "name": "Linear digest",
            "connector_execution": {
                "provider_id": "linear",
                "grant_id": "grant-1",
                "grant_token": "opaque-task-grant",
            },
        }
    )

    assert normalized["connector_execution"]["authorization_state"] == "authorized"
    assert "grant_token" not in normalized["connector_execution"]
