"""Unit coverage for Cron-only profile-local Skill App operations."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.session_context import _VAR_MAP
from tests.tools._profile_scope import mux_profile_scope
from tools.app_data_tool import _CAPABILITY_TIMEOUT, _READ_TIMEOUT
from tools.skill_operation_tool import (
    SKILL_OPERATION_SCHEMA,
    _bound_manifest,
    _check_skill_operation,
    _load_manifest,
    _profile_local_skill_files,
    clear_cron_manifest_snapshot,
    pop_cron_manifest_snapshot,
    push_unavailable_cron_manifest_snapshot,
    skill_operation_tool,
)


_BASE_URL = "http://127.0.0.1:18080/api/v1/internal/apps"
_SKILL = "application-create"
_LOGICAL_READ = "maintenance.inspect"
_LOGICAL_WRITE = "maintenance.apply"
_APP_SLUG = "action-dashboard"
_APP_READ = "maintenance.read"
_APP_WRITE = "maintenance.apply"
_DIGEST = "a" * 64


def test_explicit_unavailable_snapshot_fails_closed():
    token = push_unavailable_cron_manifest_snapshot("optional bridge failed")
    try:
        with pytest.raises(ValueError, match="optional bridge failed"):
            _bound_manifest()
    finally:
        pop_cron_manifest_snapshot(token)
        clear_cron_manifest_snapshot()


def _scope():
    return {
        "ZET_APP_DATA_BASE_URL": _BASE_URL,
        "ZETTLAB_AGENT_ACTION_TOKEN": "skill-operation-token",
        "ZET_AGENT_ID": "skill-operation-agent",
    }


def _manifest(operations=None):
    return {
        "schema_version": "zettlab.agent_app_operations.v1",
        "operations": operations
        or [
            {
                "name": _LOGICAL_READ,
                "mode": "read",
                "app_slug": _APP_SLUG,
                "app_operation": _APP_READ,
            },
            {
                "name": _LOGICAL_WRITE,
                "mode": "mutation",
                "app_slug": _APP_SLUG,
                "app_operation": _APP_WRITE,
            },
        ],
    }


def _write_profile(tmp_path: Path, *, disabled=False, manifest=None) -> Path:
    profile = tmp_path / "profile"
    skill_root = profile / "skills" / "common" / _SKILL
    runtime = skill_root / "runtime"
    runtime.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: application-create\n---\nCreate profile-local apps.\n",
        encoding="utf-8",
    )
    (runtime / "app_operations.json").write_text(
        json.dumps(_manifest() if manifest is None else manifest),
        encoding="utf-8",
    )
    disabled_yaml = "\n    - application-create" if disabled else " []"
    (profile / "config.yaml").write_text(
        "skills:\n"
        f"  disabled:{disabled_yaml}\n"
        "  platform_disabled:\n"
        "    cron: []\n",
        encoding="utf-8",
    )
    return profile


def test_legacy_hermes_manifest_schema_remains_supported(monkeypatch, tmp_path):
    manifest = _manifest()
    manifest["schema_version"] = "hermes.skill_app_operations.v1"
    profile = _write_profile(tmp_path, manifest=manifest)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    operations = _load_manifest(_SKILL)

    assert [operation.name for operation in operations] == [
        _LOGICAL_READ,
        _LOGICAL_WRITE,
    ]


def test_unknown_runtime_manifest_schema_fails_closed(monkeypatch, tmp_path):
    manifest = _manifest()
    manifest["schema_version"] = "unknown.agent_app_operations.v1"
    profile = _write_profile(tmp_path, manifest=manifest)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    with pytest.raises(ValueError, match="schema is unsupported"):
        _load_manifest(_SKILL)


def _write_second_runtime_skill(profile: Path) -> None:
    skill = profile / "skills" / "common" / "other-runtime"
    runtime = skill / "runtime"
    runtime.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: other-runtime\n---\nOther runtime.\n", encoding="utf-8"
    )
    (runtime / "app_operations.json").write_text(
        json.dumps(
            _manifest(
                [
                    {
                        "name": "other.apply",
                        "mode": "mutation",
                        "app_slug": "other-dashboard",
                        "app_operation": "records.apply",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )


@contextmanager
def _cron_scope(enabled=True, skills=(_SKILL,)):
    from gateway.session_context import (
        pop_cron_attached_skills,
        push_cron_attached_skills,
    )
    from tools.skill_operation_tool import (
        pop_cron_manifest_snapshot,
        push_cron_manifest_snapshot,
    )

    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    token = cron_var.set("1" if enabled else "")
    skills_token = push_cron_attached_skills(skills)
    snapshot_token = push_cron_manifest_snapshot()
    try:
        yield
    finally:
        pop_cron_manifest_snapshot(snapshot_token)
        pop_cron_attached_skills(skills_token)
        cron_var.reset(token)


class _Response:
    def __init__(self, payload, status=200):
        self.status = status
        self.raw = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit=-1):
        return self.raw if limit < 0 else self.raw[:limit]


def _sequence(seen, *responses):
    values = iter(responses)

    def open_request(request, timeout=None, **_kwargs):
        seen.append((request, timeout))
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return _Response(value)

    return open_request


def test_schema_exposes_no_target_or_execution_controls():
    parameters = SKILL_OPERATION_SCHEMA["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == {"action"}
    properties = parameters["properties"]
    assert properties["action"]["enum"] == ["capabilities", "invoke"]
    for forbidden in (
        "app_slug",
        "app_operation",
        "agent_id",
        "token",
        "url",
        "path",
        "command",
        "prompt",
        "provider",
        "model",
        "delivery_target",
    ):
        assert forbidden not in properties


def test_manifest_is_profile_local_strict_and_projects_logical_operations(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    operations = _load_manifest(_SKILL)
    assert [(item.name, item.mode) for item in operations] == [
        (_LOGICAL_READ, "read"),
        (_LOGICAL_WRITE, "mutation"),
    ]

    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()):
        result = json.loads(
            skill_operation_tool({"action": "capabilities"})
        )
    assert result["data"] == {
        "version": 1,
        "operations": [
            {"name": _LOGICAL_READ, "mode": "read"},
            {"name": _LOGICAL_WRITE, "mode": "mutation"},
        ],
    }
    assert _APP_SLUG not in json.dumps(result)
    assert _APP_READ not in json.dumps(result)


@pytest.mark.parametrize(
    "manifest",
    [
        {**_manifest(), "extra": True},
        _manifest([{**_manifest()["operations"][0], "extra": True}]),
        _manifest(
            [
                _manifest()["operations"][0],
                _manifest()["operations"][0],
            ]
        ),
        _manifest([{**_manifest()["operations"][0], "mode": "mutation"}]),
    ],
)
def test_manifest_contract_fails_closed(monkeypatch, tmp_path, manifest):
    profile = _write_profile(tmp_path, manifest=manifest)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    if manifest["operations"][0].get("mode") == "mutation" and not any(
        key == "extra" for key in manifest["operations"][0]
    ):
        # A manifest may declare a different mode; Local target-mode CAS is
        # what rejects that lie. The remaining malformed cases fail at load.
        assert _load_manifest(_SKILL)[0].mode == "mutation"
    else:
        with pytest.raises(ValueError):
            _load_manifest(_SKILL)


def test_deleted_disabled_ambiguous_and_symlinked_skills_are_unavailable(
    monkeypatch, tmp_path
):
    disabled = _write_profile(tmp_path / "disabled", disabled=True)
    monkeypatch.setenv("HERMES_HOME", str(disabled))
    with pytest.raises(ValueError, match="disabled"):
        _load_manifest(_SKILL)

    deleted = _write_profile(tmp_path / "deleted")
    manifest = deleted / "skills" / "common" / _SKILL / "runtime" / "app_operations.json"
    manifest.unlink()
    monkeypatch.setenv("HERMES_HOME", str(deleted))
    with pytest.raises(ValueError, match="missing"):
        _load_manifest(_SKILL)

    ambiguous = _write_profile(tmp_path / "ambiguous")
    original = ambiguous / "skills" / "common" / _SKILL
    duplicate = ambiguous / "skills" / "other" / _SKILL
    duplicate.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(original, duplicate)
    monkeypatch.setenv("HERMES_HOME", str(ambiguous))
    with pytest.raises(ValueError, match="ambiguous"):
        _load_manifest(_SKILL)

    symlinked = _write_profile(tmp_path / "symlinked")
    runtime = symlinked / "skills" / "common" / _SKILL / "runtime"
    real_runtime = symlinked / "real-runtime"
    runtime.rename(real_runtime)
    runtime.symlink_to(real_runtime, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(symlinked))
    with pytest.raises(ValueError, match="missing"):
        _load_manifest(_SKILL)


def test_profile_local_scan_is_bounded_and_does_not_follow_directory_symlinks(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path / "bounded")
    skills_root = profile / "skills"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: application-create\n---\nOutside.\n", encoding="utf-8"
    )
    (skills_root / "linked-category").symlink_to(outside, target_is_directory=True)

    files = _profile_local_skill_files(skills_root)
    assert all("linked-category" not in str(path) for path in files)

    monkeypatch.setattr("tools.skill_operation_tool._MAX_SKILL_SCAN_ENTRIES", 1)
    with pytest.raises(ValueError, match="scan limit"):
        _profile_local_skill_files(skills_root)


@pytest.mark.parametrize(
    "config",
    [
        {"skills": "corrupt"},
        {"skills": {"disabled": {"application-create": True}}},
        {"skills": {"disabled": ["application-create", 7]}},
        {"skills": {"disabled": [], "platform_disabled": []}},
        {
            "skills": {
                "disabled": [],
                "platform_disabled": {"cron": {"application-create": True}},
            }
        },
        {
            "skills": {
                "disabled": [],
                "platform_disabled": {"cron": ["application-create", 7]},
            }
        },
    ],
)
def test_malformed_skill_disabled_config_fails_closed(monkeypatch, tmp_path, config):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    with patch("hermes_cli.config.load_config", return_value=config), _cron_scope():
        with pytest.raises(ValueError, match="disabled"):
            _load_manifest(_SKILL)
        output = json.loads(skill_operation_tool({"action": "capabilities"}))

    assert output["error"]["code"] == "skill_operation_unavailable"


def test_legacy_scalar_disabled_config_remains_supported(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    config = {
        "skills": {
            "disabled": "another-skill",
            "platform_disabled": {"cron": "another-cron-skill"},
        }
    }

    with patch("hermes_cli.config.load_config", return_value=config):
        assert _load_manifest(_SKILL)


def test_profile_local_skill_lookup_failure_denies_without_raising(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.skill_operation_tool.get_skills_dir",
        side_effect=RuntimeError("scope lookup failed"),
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _check_skill_operation() is False
        output = json.loads(skill_operation_tool({"action": "capabilities"}))
    assert output["error"]["code"] == "skill_operation_unavailable"
    urlopen.assert_not_called()


def test_tool_is_cron_only_and_rechecks_delegated_scope(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    with mux_profile_scope(monkeypatch, _scope()):
        with _cron_scope(False):
            assert _check_skill_operation() is False
            denied = json.loads(
                skill_operation_tool({"action": "capabilities"})
            )
        assert denied["error"]["code"] == "cron_scope_required"

        with _cron_scope(), patch(
            "agent.delegation_context.is_delegated_child_context",
            return_value=True,
        ):
            assert _check_skill_operation() is False
            denied = json.loads(
                skill_operation_tool({"action": "capabilities"})
            )
        assert denied["error"]["code"] == "delegated_child_scope_denied"

    assert getattr(_check_skill_operation, "_profile_scope_sensitive") is True
    assert getattr(_check_skill_operation, "_session_scope_sensitive") is True


def test_job_skill_binding_cannot_be_forged_or_borrow_another_manifest(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    _write_second_runtime_skill(profile)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    with _cron_scope(skills=(_SKILL,)), mux_profile_scope(
        monkeypatch, _scope()
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        forged = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "skill": "other-runtime",
                    "operation": "other.apply",
                    "idempotency_key": "forged:1",
                }
            )
        )
        unbound = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": "other.apply",
                    "idempotency_key": "forged:2",
                }
            )
        )
    assert forged["error"]["code"] == "invalid_request"
    assert unbound["error"]["code"] == "operation_not_declared"
    urlopen.assert_not_called()

    with _cron_scope(skills=(_SKILL, "other-runtime")), mux_profile_scope(
        monkeypatch, _scope()
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        ambiguous = json.loads(skill_operation_tool({"action": "capabilities"}))
    assert ambiguous["error"]["code"] == "skill_operation_unavailable"
    assert "exactly one" in ambiguous["error"]["message"]
    urlopen.assert_not_called()

    plain_skill = profile / "skills" / "common" / "plain-helper"
    plain_skill.mkdir(parents=True)
    (plain_skill / "SKILL.md").write_text(
        "---\nname: plain-helper\n---\nNo runtime authority.\n", encoding="utf-8"
    )
    with _cron_scope(skills=(_SKILL, "plain-helper")), mux_profile_scope(
        monkeypatch, _scope()
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        mixed = json.loads(skill_operation_tool({"action": "capabilities"}))
    assert mixed["ok"] is True
    assert [item["name"] for item in mixed["data"]["operations"]] == [
        _LOGICAL_READ,
        _LOGICAL_WRITE,
    ]
    urlopen.assert_not_called()

    (plain_skill / "SKILL.md").unlink()
    with _cron_scope(skills=(_SKILL, "plain-helper")), mux_profile_scope(
        monkeypatch, _scope()
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        missing_companion = json.loads(
            skill_operation_tool({"action": "capabilities"})
        )
    assert missing_companion["error"]["code"] == "skill_operation_unavailable"
    assert "plain-helper" in missing_companion["error"]["message"]
    urlopen.assert_not_called()

    (plain_skill / "SKILL.md").write_text(
        "---\nname: plain-helper\n---\nNo runtime authority.\n", encoding="utf-8"
    )
    disabled_config = {
        "skills": {
            "disabled": ["plain-helper"],
            "platform_disabled": {"cron": []},
        }
    }
    with patch("hermes_cli.config.load_config", return_value=disabled_config), _cron_scope(
        skills=(_SKILL, "plain-helper")
    ), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        disabled_companion = json.loads(
            skill_operation_tool({"action": "capabilities"})
        )
    assert disabled_companion["error"]["code"] == "skill_operation_unavailable"
    assert "disabled" in disabled_companion["error"]["message"]
    urlopen.assert_not_called()

    with _cron_scope(skills=tuple(f"skill-{index:02d}" for index in range(17))), mux_profile_scope(
        monkeypatch, _scope()
    ):
        oversized = json.loads(skill_operation_tool({"action": "capabilities"}))
    assert oversized["error"]["code"] == "skill_operation_unavailable"

    with _cron_scope(skills=()), mux_profile_scope(
        monkeypatch, _scope()
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        missing = json.loads(skill_operation_tool({"action": "capabilities"}))
    assert missing["error"]["code"] == "skill_operation_unavailable"
    urlopen.assert_not_called()


@pytest.mark.parametrize("tamper", ["rewrite", "delete"])
def test_run_start_manifest_snapshot_rejects_same_run_tampering(
    monkeypatch, tmp_path, tamper
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    manifest_path = (
        profile
        / "skills"
        / "common"
        / _SKILL
        / "runtime"
        / "app_operations.json"
    )

    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        if tamper == "delete":
            manifest_path.unlink()
        else:
            manifest_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "name": _LOGICAL_WRITE,
                                "mode": "mutation",
                                "app_slug": "other-dashboard",
                                "app_operation": "records.apply",
                            }
                        ]
                    )
                ),
                encoding="utf-8",
            )
        output = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": _LOGICAL_WRITE,
                    "idempotency_key": "same-run-tamper:1",
                }
            )
        )

    assert output["error"]["code"] == "skill_operation_unavailable"
    assert "changed" in output["error"]["message"]
    urlopen.assert_not_called()


def test_invoke_uses_only_manifest_target_and_requires_matching_mode(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    seen = []
    capabilities = {
        "version": 1,
        "operations": [
            {"name": _APP_READ, "mode": "read"},
            {"name": _APP_WRITE, "mode": "mutation"},
        ],
        "capability_digest": _DIGEST,
    }
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(seen, capabilities, {"items": [{"id": "entry-1"}]}),
    ):
        output = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": _LOGICAL_READ,
                    "query": {"limit": "2"},
                    "payload": {
                        "record": {"source_url": "https://example.com/item"}
                    },
                }
            )
        )
    assert output["ok"] is True
    assert [item[1] for item in seen] == [
        _CAPABILITY_TIMEOUT,
        _READ_TIMEOUT,
    ]
    capability_request, invoke_request = [item[0] for item in seen]
    assert capability_request.full_url == f"{_BASE_URL}/{_APP_SLUG}/capabilities"
    assert invoke_request.full_url == (
        f"{_BASE_URL}/{_APP_SLUG}/operations/{_APP_READ}"
    )
    assert json.loads(invoke_request.data) == {
        "capability_digest": _DIGEST,
        "query": {"limit": "2"},
        "payload": {
            "record": {"source_url": "https://example.com/item"}
        },
    }

    seen = []
    wrong_mode = {
        **capabilities,
        "operations": [{"name": _APP_READ, "mode": "mutation"}],
    }
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen", _sequence(seen, wrong_mode)
    ):
        mismatch = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": _LOGICAL_READ,
                }
            )
        )
    assert mismatch["error"]["code"] == "operation_contract_mismatch"
    assert len(seen) == 1


def test_read_invoke_does_not_retry_outside_local_server(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    seen = []
    capabilities = {
        "version": 1,
        "operations": [{"name": _APP_READ, "mode": "read"}],
        "capability_digest": _DIGEST,
    }
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(seen, capabilities, TimeoutError("unknown read outcome")),
    ):
        output = json.loads(
            skill_operation_tool(
                {"action": "invoke", "operation": _LOGICAL_READ}
            )
        )

    assert output["error"]["code"] == "transport_error"
    assert [item[1] for item in seen] == [
        _CAPABILITY_TIMEOUT,
        _READ_TIMEOUT,
    ]


@pytest.mark.parametrize(
    "field",
    [
        "access_key",
        "apiKey",
        "bearer",
        "credential",
        "password",
        "secret_key",
        "token",
        "headers",
        "prompt",
        "provider",
        "model",
        "file_path",
        "callback_url",
        "callback_urls",
        "endpoint_uris",
        "file_paths",
        "source_urls",
    ],
)
def test_sensitive_payload_never_reaches_transport(
    monkeypatch, tmp_path, field
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        output = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": _LOGICAL_READ,
                    "payload": {"nested": {field: "do-not-forward"}},
                }
            )
        )
    assert output["error"]["code"] == "invalid_request"
    urlopen.assert_not_called()


def test_mutation_requires_idempotency_key_before_transport(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    with _cron_scope(), mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        output = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": _LOGICAL_WRITE,
                    "payload": {"revision": 1},
                }
            )
        )
    assert output["error"]["code"] == "invalid_request"
    urlopen.assert_not_called()
