"""Flow coverage for Cron-bound profile-local Skill operations."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import cron.scheduler as cron_scheduler
from gateway.session_context import (
    cron_attached_skills,
    pop_cron_attached_skills,
    push_cron_attached_skills,
    reset_session_vars,
)
from tools.skill_operation_tool import _check_skill_operation, skill_operation_tool


class _DummySessionDB:
    def set_session_title(self, *_args, **_kwargs):
        pass

    def end_session(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


def _write_runtime_skill(
    profile: Path, *, skill: str, logical_operation: str, app_slug: str
) -> None:
    skill_root = profile / "skills" / "common" / skill
    runtime_root = skill_root / "runtime"
    runtime_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        f"name: {skill}\n"
        f"description: Runtime operation scope for {skill}.\n"
        "---\n"
        f"Handle only {skill} operations.\n",
        encoding="utf-8",
    )
    (runtime_root / "app_operations.json").write_text(
        json.dumps(
            {
                "schema_version": "hermes.skill_app_operations.v1",
                "operations": [
                    {
                        "name": logical_operation,
                        "mode": "read",
                        "app_slug": app_slug,
                        "app_operation": "records.read",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_job_isolates_concurrent_skill_manifests_and_restores_after_failure(
    monkeypatch, tmp_path
):
    """``run_job`` owns binding, isolation, and finally restoration."""
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "skills:\n"
        "  disabled: []\n"
        "  platform_disabled:\n"
        "    cron: []\n",
        encoding="utf-8",
    )
    jobs = {
        "runtime-alpha": {
            "id": "scope-alpha",
            "name": "Alpha scope",
            "prompt": "RUN_ALPHA_SCOPE",
            "skills": ["runtime-alpha"],
            "schedule_display": "manual",
        },
        "runtime-beta": {
            "id": "scope-beta",
            "name": "Beta scope",
            "prompt": "RUN_BETA_SCOPE",
            "skills": ["runtime-beta"],
            "schedule_display": "manual",
        },
    }
    expected_operations = {
        "runtime-alpha": "alpha.inspect",
        "runtime-beta": "beta.inspect",
    }
    _write_runtime_skill(
        profile,
        skill="runtime-alpha",
        logical_operation=expected_operations["runtime-alpha"],
        app_slug="alpha-app",
    )
    _write_runtime_skill(
        profile,
        skill="runtime-beta",
        logical_operation=expected_operations["runtime-beta"],
        app_slug="beta-app",
    )

    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setenv(
        "ZET_APPHOST_BASE_URL", "http://127.0.0.1:19090/api/v1/internal/apps"
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "test-action-token")
    monkeypatch.setenv("ZET_AGENT_ID", "scope-agent")
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: profile)
    monkeypatch.setattr(
        cron_scheduler, "_refresh_cron_dotenv_for_legacy_process", lambda: None
    )
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(cron_scheduler, "_guard_job_credential_exfil", lambda _job: None)
    monkeypatch.setattr("hermes_state.SessionDB", _DummySessionDB)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "",
            "requested_provider": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv", lambda **_kwargs: None
    )
    monkeypatch.setattr("hermes_cli.env_loader.reset_secret_source_cache", lambda: None)
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])

    barrier = threading.Barrier(2)
    observed: dict[str, dict[str, object]] = {}
    observed_lock = threading.Lock()

    class _ScopedAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_conversation(self, prompt, **_kwargs):
            skill = "runtime-alpha" if "RUN_ALPHA_SCOPE" in prompt else "runtime-beta"
            barrier.wait(timeout=5)
            capabilities = json.loads(
                skill_operation_tool({"action": "capabilities"})
            )
            with observed_lock:
                observed[skill] = {
                    "attached": cron_attached_skills(),
                    "available": _check_skill_operation(),
                    "capabilities": capabilities,
                }
            if skill == "runtime-beta":
                raise RuntimeError("intentional beta failure")
            return {
                "completed": True,
                "failed": False,
                "final_response": "alpha complete",
                "turn_exit_reason": "",
            }

        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", _ScopedAgent)
    reset_session_vars()

    def run_with_ambient(skill: str):
        ambient = f"ambient-{skill}"
        token = push_cron_attached_skills([ambient])
        try:
            result = cron_scheduler.run_job(jobs[skill])
            restored = cron_attached_skills()
            return result, restored
        finally:
            pop_cron_attached_skills(token)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            skill: pool.submit(run_with_ambient, skill) for skill in jobs
        }
        results = {skill: future.result(timeout=15) for skill, future in futures.items()}

    for skill, operation in expected_operations.items():
        assert observed[skill]["attached"] == (skill,)
        assert observed[skill]["available"] is True
        assert observed[skill]["capabilities"] == {
            "ok": True,
            "untrusted_app_data": True,
            "data": {
                "version": 1,
                "operations": [{"name": operation, "mode": "read"}],
            },
        }
        assert results[skill][1] == (f"ambient-{skill}",)

    alpha_result, _alpha_restored = results["runtime-alpha"]
    beta_result, _beta_restored = results["runtime-beta"]
    assert alpha_result[0] is True
    assert alpha_result[2] == "alpha complete"
    assert beta_result[0] is False
    assert "intentional beta failure" in (beta_result[3] or "")
    assert cron_attached_skills() == ()
