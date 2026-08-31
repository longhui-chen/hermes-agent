from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

import pytest
from jsonschema import Draft7Validator

from plugins.video_edit import client, normalizer, preferences, schemas, state, tools


EXPECTED_BUSINESS_PROPERTIES = {
    "video_edit_preferences_resolve": {"task_id", "scene", "preferences", "silent"},
    "video_edit_preferences_update": {"scope", "scene", "kind", "action", "preferences"},
    "video_edit_preferences_record_success": {"scene", "preferences", "confirmed_fields"},
    "video_edit_upload_assets": {"workflow_id", "files", "normalize"},
    "video_edit_create_project": {"workflow_id", "user_prompt"},
    "video_edit_wait_project": {"workflow_id", "max_wait_seconds"},
    "video_edit_download_result": {"workflow_id", "filename"},
    "video_edit_proactive_resolve": {"manifest_id", "task_id"},
    "video_edit_proactive_report": {"workflow_id"},
}

EXPECTED_REQUIRED = {
    "video_edit_preferences_resolve": [],
    "video_edit_preferences_update": ["scope", "kind", "action"],
    "video_edit_preferences_record_success": ["preferences"],
    "video_edit_upload_assets": ["workflow_id"],
    "video_edit_create_project": ["workflow_id"],
    "video_edit_wait_project": ["workflow_id"],
    "video_edit_download_result": ["workflow_id"],
    "video_edit_proactive_resolve": ["manifest_id"],
    "video_edit_proactive_report": ["workflow_id"],
}

HELP_SCHEMA_DIGEST_BY_VERSION = {
    # Update this snapshot only together with an intentional HELP_SCHEMA_VERSION bump.
    "1.0": "8b529ed56597d24dc8e31646f44c15c5142d7293e1bb3359490638073824da45",
    "1.1": "dbbc6daf1d45826ca7ea106ed0125899bb856893fca721895c434e5048b696d5",
    "1.2": "e9b203e3ec8c9ecd947b2c5cdc8ad26cdc676c4c60249b3fb550c91ab9d032e5",
    "1.3": "2faf5826132fb279818e7b8e35779b70a776a7ef3afe45b49979d3b1d88a04a9",
    "1.4": "6e0bb9af2ccfeb04a1890a3595a5690b152a849d46ed67b2815906f7a422cc9d",
    "1.5": "57a5b21662652e76059efdec848c1b55500c9463fbdc82a363489003aacaa541",
    "1.6": "e8fa7a6778e0f60d8690db7a6ae4ebf0a7dfb3f7f8b1b8c77271644cdcf62709",
    "1.7": "baccd822ed985b1b0b69008be51083bc1009d86060fc12a76d792d19dcd1ff08",
    "1.8": "b9ecbee9306e6cd0213d59e22bee78cb824a537ac55057148fc55685685725a2",
}

OVERVIEW_FIELDS = {
    "tool",
    "schema_version",
    "purpose",
    "when_to_use",
    "side_effects",
    "required_inputs",
    "optional_inputs",
    "defaults",
    "enums_and_ranges",
    "cross_field_invariants",
    "reusable_business_ids",
    "success_outputs",
    "recoverable_errors",
    "terminal_errors",
    "retryable",
    "next_tools",
    "minimal_valid_call",
    "common_mistake",
    "corrected_call",
    "bad_recovery",
    "forbidden_fallbacks",
    "ignored_business_fields",
}


class _UnreadableContextValue:
    def __str__(self) -> str:  # pragma: no cover - called only on a regression
        raise AssertionError("help must not inspect handler context")


def _install_side_effect_sentinels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def bomb(label: str):
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(label)
            raise AssertionError(f"unexpected help side effect: {label}")

        return fail

    targets = (
        (tools, "agent_id_from_kwargs"),
        (tools, "validate_input_file"),
        (tools, "validate_output_file"),
        (tools, "result_path"),
        (state, "get"),
        (state, "update"),
        (state, "workflow_id"),
        (preferences, "resolve"),
        (preferences, "update"),
        (preferences, "record_success"),
        (client, "upload"),
        (client, "create_project"),
        (client, "poll_project"),
        (client, "download"),
        (client, "proactive_resolve"),
        (client, "proactive_report"),
        (client, "file_evidence"),
        (normalizer, "normalize_files"),
        (normalizer, "cleanup"),
        (normalizer, "_run_bounded_subprocess"),
    )
    for owner, attribute in targets:
        monkeypatch.setattr(owner, attribute, bomb(f"{owner.__name__}.{attribute}"))
    return calls


def _install_resolve_boundary_trace(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def agent_id_from_context(_kwargs: dict[str, Any]) -> str:
        calls.append("agent_context")
        return "agent-help-trace"

    def workflow_id(task_id: str, agent_id: str) -> str:
        calls.append("state.workflow_id")
        assert task_id == "TRACE_TASK_ID"
        assert agent_id == "agent-help-trace"
        return "trace-workflow-id"

    def resolve(
        agent_id: str,
        scene: str,
        supplied: dict[str, Any] | None,
        *,
        silent: bool,
    ) -> dict[str, Any]:
        calls.append("preferences.resolve")
        assert agent_id == "agent-help-trace"
        assert scene == "general"
        assert supplied is None
        assert silent is False
        return {
            "scene": scene,
            "preferences": {},
            "sources": {},
            "memory_hit": False,
        }

    def create_or_validate_identity(
        workflow_id: str,
        agent_id: str,
        identity: dict[str, Any],
        initial: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append("state.create_or_validate_identity")
        assert workflow_id == "trace-workflow-id"
        assert agent_id == "agent-help-trace"
        assert identity["resolve_request"]["task_id"] == "TRACE_TASK_ID"
        assert initial["status"] == "preferences_resolved"
        assert kwargs["legacy_requires_non_proactive"] is True
        return copy.deepcopy({"workflow_id": workflow_id, **identity, **initial})

    monkeypatch.setattr(tools, "agent_id_from_kwargs", agent_id_from_context)
    monkeypatch.setattr(state, "workflow_id", workflow_id)
    monkeypatch.setattr(preferences, "resolve", resolve)
    monkeypatch.setattr(state, "create_or_validate_identity", create_or_validate_identity)
    return calls


def _register_resolve_handler() -> None:
    import model_tools
    from tools.registry import registry

    name = "video_edit_preferences_resolve"
    registry.register(
        name=name,
        toolset="video_edit",
        schema=schemas.TOOL_DEFINITIONS_BY_NAME[name],
        handler=tools.HANDLERS[name],
    )
    model_tools._clear_tool_defs_cache()


def _walk_schema_contract(
    value: dict[str, Any],
    path: str,
    defaults: dict[str, Any],
    constraints: dict[str, Any],
) -> None:
    if "default" in value:
        defaults[path] = copy.deepcopy(value["default"])
    selected = {
        key: copy.deepcopy(value[key])
        for key in (
            "enum",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "minProperties",
            "maxProperties",
            "pattern",
        )
        if key in value
    }
    if selected:
        constraints[path] = selected
    for field, child in (value.get("properties") or {}).items():
        _walk_schema_contract(child, f"{path}.{field}" if path else field, defaults, constraints)
    if isinstance(value.get("items"), dict):
        _walk_schema_contract(value["items"], f"{path}[]", defaults, constraints)


def _parsed(handler, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return json.loads(handler(args, **kwargs))


def _natural_language_values(value: Any, field: str = "") -> list[str]:
    """Collect prose and examples while excluding stable protocol identifiers."""
    if isinstance(value, dict):
        return [
            text
            for key, child in value.items()
            if key not in {"code", "next_tool", "reason_code", "tool", "when"}
            for text in _natural_language_values(child, key)
        ]
    if isinstance(value, list):
        return [text for child in value for text in _natural_language_values(child, field)]
    return [value] if isinstance(value, str) else []


def test_nine_tools_keep_business_schema_and_add_only_unified_help_controls():
    assert len(schemas.TOOL_DEFINITIONS) == 9
    assert set(schemas.TOOL_DEFINITIONS_BY_NAME) == set(EXPECTED_BUSINESS_PROPERTIES)
    assert set(tools.HANDLERS) == set(EXPECTED_BUSINESS_PROPERTIES)
    assert set(schemas.TOOL_HELP_METADATA) == set(EXPECTED_BUSINESS_PROPERTIES)
    assert len({handler.__code__ for handler in tools.HANDLERS.values()}) == 1

    forbidden_permission_fields = {
        "permission",
        "permissions",
        "authorization",
        "approval",
        "grant",
        "receipt",
        "capability",
        "credential",
    }
    for definition in schemas.TOOL_DEFINITIONS:
        name = definition["name"]
        parameters = definition["parameters"]
        properties = parameters["properties"]
        assert parameters.get("required", []) == []
        assert schemas.business_required_names(name) == EXPECTED_REQUIRED[name]
        assert parameters["if"] == {
            "properties": {"help": {"const": True}},
            "required": ["help"],
        }
        assert parameters["then"] == {}
        assert parameters["else"] == {"required": EXPECTED_REQUIRED[name]}
        if name == "video_edit_preferences_update":
            assert len(parameters["allOf"]) == 2
        else:
            assert "allOf" not in parameters
        assert set(properties) == EXPECTED_BUSINESS_PROPERTIES[name] | {"help", "help_topic"}
        assert properties["help"] == {
            "type": "boolean",
            "default": False,
            "description": "Return this tool's side-effect-free contract instead of executing it.",
        }
        assert properties["help_topic"]["type"] == "string"
        assert properties["help_topic"]["enum"] == list(schemas.HELP_TOPICS)
        assert properties["help_topic"]["default"] == "overview"
        assert not forbidden_permission_fields.intersection(properties)


def test_help_schema_version_is_bound_to_canonical_tool_schema_snapshot():
    canonical = json.dumps(
        {
            "definitions": schemas.TOOL_DEFINITIONS,
            "rendered_help": {
                name: {
                    topic: schemas.render_tool_help(
                        name,
                        {"help": True, "help_topic": topic},
                    )
                    for topic in (*schemas.HELP_TOPICS, "NOT_A_TOPIC")
                }
                for name in schemas.TOOL_DEFINITIONS_BY_NAME
            },
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()

    assert schemas.HELP_SCHEMA_VERSION in HELP_SCHEMA_DIGEST_BY_VERSION
    assert digest == HELP_SCHEMA_DIGEST_BY_VERSION[schemas.HELP_SCHEMA_VERSION]


def test_all_help_only_mixed_and_unknown_topic_calls_are_side_effect_free(monkeypatch):
    calls = _install_side_effect_sentinels(monkeypatch)
    unreadable = _UnreadableContextValue()

    for name, handler in tools.HANDLERS.items():
        overview = _parsed(handler, {"help": True}, agent_id=unreadable, task_id=unreadable)
        assert OVERVIEW_FIELDS.issubset(overview)
        assert overview["tool"] == name
        assert overview["side_effects"] == "none"
        assert overview["ignored_business_fields"] == []

        business_field = next(iter(EXPECTED_BUSINESS_PROPERTIES[name]))
        mixed = _parsed(
            handler,
            {"help": True, business_field: unreadable},
            agent_id=unreadable,
            task_id=unreadable,
        )
        assert mixed["side_effects"] == "none"
        assert mixed["ignored_business_fields"] == [business_field]

        unknown = _parsed(
            handler,
            {
                "help": True,
                "help_topic": "NOT_A_TOPIC",
                "unknown_sensitive_key": unreadable,
            },
            agent_id=unreadable,
        )
        assert unknown["error"] == "help_topic is not supported"
        assert unknown["help_topic"] == "invalid"
        assert unknown["ignored_business_fields"] == []
        assert unknown["allowed_topics"] == list(schemas.HELP_TOPICS)
        assert "NOT_A_TOPIC" not in json.dumps(unknown)
        assert "unknown_sensitive_key" not in json.dumps(unknown)

    assert calls == []


@pytest.mark.parametrize("topic", schemas.HELP_TOPICS)
def test_every_help_topic_is_available_for_every_tool(topic):
    for name, handler in tools.HANDLERS.items():
        result = _parsed(handler, {"help": True, "help_topic": topic})
        assert result["tool"] == name
        assert result["help_topic"] == topic
        assert result["schema_version"] == schemas.HELP_SCHEMA_VERSION
        assert re.fullmatch(r"\d+\.\d+", result["schema_version"])
        assert result["side_effects"] == "none"
        if topic == "examples":
            assert result["examples"]
            assert all(set(example) == {"tool", "arguments"} for example in result["examples"])


def test_overview_technical_contract_is_derived_from_each_registered_schema():
    for definition in schemas.TOOL_DEFINITIONS:
        name = definition["name"]
        parameters = definition["parameters"]
        properties = parameters["properties"]
        overview = _parsed(tools.HANDLERS[name], {"help": True})

        defaults: dict[str, Any] = {}
        constraints: dict[str, Any] = {}
        for field, field_schema in properties.items():
            _walk_schema_contract(field_schema, field, defaults, constraints)

        required_names = schemas.business_required_names(name)
        assert list(overview["required_inputs"]) == required_names
        assert overview["required_inputs"] == {
            field: properties[field] for field in required_names
        }
        assert overview["optional_inputs"] == {
            field: schema
            for field, schema in properties.items()
            if field not in required_names
        }
        assert overview["defaults"] == defaults
        assert overview["enums_and_ranges"] == constraints
        assert overview["purpose"] == definition["description"]
        assert set(overview["reusable_business_ids"]).issubset(EXPECTED_BUSINESS_PROPERTIES[name])
        assert schemas.validate_tool_arguments(
            name, overview["minimal_valid_call"]["arguments"]
        ) == []
        assert schemas.validate_tool_arguments(
            name, overview["corrected_call"]["arguments"]
        ) == []


def test_preference_update_help_preserves_unrepresentable_qualifiers():
    overview = _parsed(
        tools.HANDLERS["video_edit_preferences_update"],
        {"help": True},
    )
    invariants = " ".join(overview["cross_field_invariants"]).lower()

    assert "exact semantic scene category" in invariants
    assert "cannot encode arbitrary predicates" in invariants
    assert "never drop qualifiers" in invariants
    assert "reinterpret a qualifier as scene" in invariants
    assert "do not call this tool" in invariants


def test_sanitized_model_schemas_keep_all_normal_required_inputs_visible():
    from tools.schema_sanitizer import sanitize_tool_schemas

    visible = sanitize_tool_schemas(
        [
            {"type": "function", "function": copy.deepcopy(definition)}
            for definition in schemas.TOOL_DEFINITIONS
        ]
    )

    assert len(visible) == 9
    for tool_definition in visible:
        function = tool_definition["function"]
        name = function["name"]
        parameters = function["parameters"]
        required = EXPECTED_REQUIRED[name]
        validator = Draft7Validator(parameters)
        assert "allOf" not in parameters
        assert parameters.get("required", []) == []
        assert parameters["if"] == {
            "properties": {"help": {"const": True}},
            "required": ["help"],
        }
        assert parameters["then"] == {}
        assert parameters["else"] == {"required": required}
        if required:
            assert not validator.is_valid({})
            assert not validator.is_valid({"help": False})
        else:
            assert validator.is_valid({})
            assert validator.is_valid({"help": False})
        assert validator.is_valid({"help": True})
        normal_arguments = schemas.TOOL_HELP_METADATA[name]["minimal_valid_call"][
            "arguments"
        ]
        assert validator.is_valid(normal_arguments)
        for field in required:
            missing_required = copy.deepcopy(normal_arguments)
            missing_required.pop(field)
            assert not validator.is_valid(missing_required)
        assert (
            f"Normal-call required inputs: {', '.join(required) or 'none'}"
            in parameters["description"]
        )
        for field in required:
            assert "Required for normal calls" in parameters["properties"][field][
                "description"
            ]


def test_manual_recovery_metadata_matches_handlers_and_state_machine(monkeypatch):
    known_tools = set(tools.HANDLERS)
    monkeypatch.setattr(
        tools,
        "agent_id_from_kwargs",
        lambda _kwargs: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    for name, metadata in schemas.TOOL_HELP_METADATA.items():
        errors = metadata["recoverable_errors"] + metadata["terminal_errors"]
        reason_codes = [
            item["reason_code"]
            for item in errors
        ]
        assert len(reason_codes) == len(set(reason_codes))
        assert "invalid_arguments" in reason_codes
        assert "invalid_help_topic" in reason_codes
        assert "local_state_unavailable" in reason_codes
        assert "transient_failure" in reason_codes
        assert all(item["code"] for item in errors)
        for transition in metadata["next_tools"]:
            assert transition["tool"] is None or transition["tool"] in known_tools
        assert metadata["minimal_valid_call"]["tool"] == name
        assert metadata["corrected_call"]["tool"] == name

        result = _parsed(
            tools.HANDLERS[name],
            metadata["minimal_valid_call"]["arguments"],
        )
        contract = schemas.error_contract(name, "transient_failure")
        assert result["code"] == schemas.business_failure_code(name)
        assert result["reason_code"] == contract["reason_code"]
        assert result["retryable"] == contract["retryable"]
        assert result["next"] == contract["next_tool"]
        assert result["recovery"] == contract["recovery"]

    contract = schemas.error_contract("video_edit_wait_project", "project_terminal")
    terminal = json.loads(
        tools._terminal_fail(
            "video_edit_wait_project",
            "project_terminal",
            "forced terminal state",
        )
    )
    assert terminal["code"] == contract["code"]
    assert terminal["reason_code"] == contract["reason_code"]


def test_every_runtime_error_response_matches_its_help_recovery_contract():
    for name, metadata in schemas.TOOL_HELP_METADATA.items():
        for contract in metadata["recoverable_errors"] + metadata["terminal_errors"]:
            result = _parsed(
                lambda _args: tools._failure(
                    name,
                    contract["reason_code"],
                    "bounded test error",
                ),
                {},
            )
            assert result["code"] == contract["code"]
            assert result["reason_code"] == contract["reason_code"]
            assert result["retryable"] == contract["retryable"]
            assert result["next"] == contract["next_tool"]
            assert result["recovery"] == contract["recovery"]


def test_wait_help_declares_missing_completed_result_recovery():
    contract = schemas.error_contract(
        "video_edit_wait_project",
        "project_not_completed",
    )

    assert contract["retryable"] is True
    assert contract["next_tool"] == "video_edit_wait_project"


def test_preferences_resolve_help_declares_terminal_workflow_conflict():
    contract = schemas.error_contract(
        "video_edit_preferences_resolve",
        "workflow_unavailable",
    )

    assert contract["code"] == "preferences_resolve_failed"
    assert contract["retryable"] is False
    assert contract["next_tool"] is None


def test_terminal_help_recovery_wording_is_business_neutral():
    admission_recovery = (
        "Stop. Surface the video service admission failure without retrying or "
        "bypassing the plugin."
    )
    workflow_recovery = (
        "Stop. Surface the unavailable trusted workflow checkpoint without retrying "
        "or bypassing the plugin."
    )
    admission_entries = []
    workflow_entries = []
    for metadata in schemas.TOOL_HELP_METADATA.values():
        for item in metadata["terminal_errors"]:
            if item["reason_code"] == "service_admission_failed":
                admission_entries.append(item)
            elif item["reason_code"] == "workflow_unavailable":
                workflow_entries.append(item)

    assert admission_entries
    assert workflow_entries
    assert {item["recovery"] for item in admission_entries} == {admission_recovery}
    assert {item["recovery"] for item in workflow_entries} == {workflow_recovery}
    rendered = " ".join(
        item["recovery"] for item in admission_entries + workflow_entries
    ).lower()
    assert not any(
        forbidden in rendered
        for forbidden in (
            "authorization",
            "credential",
            "permission",
            "profile",
            "scope",
            "token",
        )
    )


@pytest.mark.parametrize(
    ("name", "arguments", "rule"),
    [
            ("video_edit_preferences_resolve", {"task_id": 1}, "type"),
        (
            "video_edit_preferences_update",
            {"scope": "invalid", "kind": "soft", "action": "set"},
            "enum",
        ),
        ("video_edit_preferences_record_success", {"preferences": []}, "type"),
        (
            "video_edit_upload_assets",
            {"workflow_id": "WORKFLOW_ID", "files": ["MEDIA_REFERENCE"] * 9},
            "maxItems",
        ),
        (
            "video_edit_create_project",
            {"workflow_id": "WORKFLOW_ID", "user_prompt": "x" * 513},
            "maxLength",
        ),
        (
            "video_edit_wait_project",
            {"workflow_id": "WORKFLOW_ID", "max_wait_seconds": 481},
            "maximum",
        ),
        (
            "video_edit_download_result",
            {"workflow_id": "WORKFLOW_ID", "unexpected": True},
            "additionalProperties",
        ),
        (
            "video_edit_proactive_resolve",
            {"manifest_id": 1, "task_id": "TASK_ID"},
            "type",
        ),
        ("video_edit_proactive_report", {"help": False}, "required"),
    ],
)
def test_invalid_normal_calls_fail_before_context_state_files_or_network(
    monkeypatch, name, arguments, rule
):
    calls = _install_side_effect_sentinels(monkeypatch)
    result = _parsed(
        tools.HANDLERS[name],
        arguments,
        agent_id=_UnreadableContextValue(),
    )
    assert result["code"] == "invalid_arguments"
    assert rule in {issue["rule"] for issue in result["issues"]}
    assert calls == []


def test_schema_validator_accepts_boundaries_and_rejects_every_constraint_family():
    accepted = (
        (
            "video_edit_preferences_resolve",
            {
                "task_id": "TASK_ID",
                "preferences": {
                    "duration": 1,
                    "aspect_ratio": "9:16",
                    "editing_directives": [
                        "energetic_pacing",
                        "keep_original_audio",
                        "add_captions",
                    ],
                },
            },
        ),
        (
            "video_edit_wait_project",
            {"workflow_id": "WORKFLOW_ID", "max_wait_seconds": 480.0},
        ),
        (
            "video_edit_upload_assets",
            {"workflow_id": "WORKFLOW_ID", "files": ["MEDIA_REFERENCE"] * 8},
        ),
        (
            "video_edit_preferences_resolve",
            {
                "task_id": "x" * 256,
                "preferences": {
                    "style": "x" * preferences.MAX_STYLE_LENGTH,
                    "editing_directives": ["preserve_dialogue"],
                },
            },
        ),
    )
    for name, arguments in accepted:
        assert schemas.validate_tool_arguments(name, arguments) == []

    rejected = (
        (
            "video_edit_preferences_resolve",
            {"task_id": "TASK_ID", "preferences": {"duration": 0}},
            "minimum",
        ),
        (
            "video_edit_preferences_resolve",
            {"task_id": "TASK_ID", "preferences": {"aspect_ratio": "invalid"}},
            "enum",
        ),
        (
            "video_edit_preferences_resolve",
            {
                "task_id": "TASK_ID",
                "preferences": {"editing_directives": ["a", "b", "c", "d"]},
            },
            "maxItems",
        ),
        (
            "video_edit_preferences_resolve",
            {
                "task_id": "TASK_ID",
                "preferences": {"editing_directives": ["unsupported_directive"]},
            },
            "enum",
        ),
        (
            "video_edit_preferences_resolve",
            {
                "task_id": "TASK_ID",
                "preferences": {"style": "x" * (preferences.MAX_STYLE_LENGTH + 1)},
            },
            "maxLength",
        ),
        (
            "video_edit_upload_assets",
            {"workflow_id": "WORKFLOW_ID", "files": []},
            "minItems",
        ),
        (
            "video_edit_proactive_resolve",
            {"manifest_id": "MANIFEST_ID", "task_id": "x" * 257},
            "maxLength",
        ),
        (
            "video_edit_wait_project",
            {"workflow_id": "WORKFLOW_ID", "max_wait_seconds": 14},
            "minimum",
        ),
        (
            "video_edit_upload_assets",
            {"workflow_id": "WORKFLOW_ID", "normalize": "true"},
            "type",
        ),
        (
            "video_edit_preferences_update",
            {"scope": "global", "kind": "soft", "action": "set"},
            "required",
        ),
        (
            "video_edit_preferences_update",
            {
                "scope": "scene",
                "kind": "soft",
                "action": "set",
                "preferences": {"style": "travel"},
            },
            "required",
        ),
        (
            "video_edit_preferences_update",
            {
                "scope": "global",
                "kind": "soft",
                "action": "set",
                "preferences": {"duraton": 30},
            },
            "additionalProperties",
        ),
        (
            "video_edit_preferences_update",
            {
                "scope": "global",
                "kind": "soft",
                "action": "set",
                "preferences": {},
            },
            "minProperties",
        ),
    )
    for name, arguments, expected_rule in rejected:
        issues = schemas.validate_tool_arguments(name, arguments)
        assert expected_rule in {issue["rule"] for issue in issues}


def test_preference_schema_reuses_runtime_cleaning_constraints():
    preference_container = schemas.TOOL_DEFINITIONS_BY_NAME[
        "video_edit_preferences_resolve"
    ]["parameters"]["properties"]["preferences"]
    preference_schema = preference_container["properties"]

    assert preference_schema["style"]["maxLength"] == preferences.MAX_STYLE_LENGTH
    assert preference_schema["user_prompt"]["maxLength"] == (
        preferences.MAX_USER_PROMPT_LENGTH
    )
    assert preference_schema["editing_directives"]["maxItems"] == (
        preferences.MAX_EDITING_DIRECTIVES
    )
    assert preference_schema["editing_directives"]["items"]["enum"] == sorted(
        preferences.VALID_DIRECTIVES
    )
    assert preference_container["additionalProperties"] is False


def test_preferences_resolve_help_keeps_full_prompt_inside_preferences():
    definition = schemas.TOOL_DEFINITIONS_BY_NAME[
        "video_edit_preferences_resolve"
    ]
    properties = definition["parameters"]["properties"]
    assert "user_prompt" not in properties
    assert "user_prompt" in properties["preferences"]["properties"]

    examples = _parsed(
        tools.handle_preferences_resolve,
        {"help": True, "help_topic": "examples"},
    )["examples"]
    corrected = examples[-1]["arguments"]
    assert "user_prompt" not in corrected
    assert corrected["preferences"]["user_prompt"] == "FULL_USER_REQUEST"

    rejected = _parsed(
        tools.handle_preferences_resolve,
        {"user_prompt": "FULL_USER_REQUEST"},
    )
    assert rejected["reason_code"] == "invalid_arguments"
    assert rejected["corrected_call"]["arguments"]["preferences"][
        "user_prompt"
    ] == "FULL_USER_REQUEST"


@pytest.mark.parametrize(
    ("handler", "arguments", "expected_ids"),
    [
        (
            tools.handle_preferences_resolve,
            {"task_id": "task-live-123", "scene": 7},
            {"task_id": "task-live-123"},
        ),
        (
            tools.handle_wait_project,
            {"workflow_id": "workflow-live-123", "max_wait_seconds": 1},
            {"workflow_id": "workflow-live-123"},
        ),
        (
            tools.handle_download_result,
            {"workflow_id": "workflow-live-123", "filename": "杭州成片.mp4"},
            {"workflow_id": "workflow-live-123"},
        ),
        (
            tools.handle_proactive_resolve,
            {
                "manifest_id": "manifest-live-123",
                "task_id": "task-live-123",
                "unexpected": True,
            },
            {
                "manifest_id": "manifest-live-123",
                "task_id": "task-live-123",
            },
        ),
    ],
)
def test_invalid_argument_recovery_preserves_valid_reusable_business_ids(
    handler,
    arguments,
    expected_ids,
):
    rejected = _parsed(handler, arguments)

    assert rejected["reason_code"] == "invalid_arguments"
    repaired = rejected["corrected_call"]["arguments"]
    for field, value in expected_ids.items():
        assert repaired[field] == value


def test_download_filename_pattern_is_schema_derived_and_enforced():
    name = "video_edit_download_result"
    filename_schema = schemas.TOOL_DEFINITIONS_BY_NAME[name]["parameters"][
        "properties"
    ]["filename"]
    assert filename_schema["pattern"] == r"^[A-Za-z0-9_.-]{1,128}$"
    assert "appends .mp4" in filename_schema["description"]

    accepted = schemas.validate_tool_arguments(
        name,
        {"workflow_id": "WORKFLOW_ID", "filename": "family_clip.mp4"},
    )
    rejected = schemas.validate_tool_arguments(
        name,
        {"workflow_id": "WORKFLOW_ID", "filename": "家庭成片.mp4"},
    )
    assert accepted == []
    assert "pattern" in {issue["rule"] for issue in rejected}

    rendered = _parsed(
        tools.handle_download_result,
        {"help": True, "help_topic": "inputs"},
    )
    assert rendered["enums_and_ranges"]["filename"]["pattern"] == (
        filename_schema["pattern"]
    )


def test_help_examples_and_contract_do_not_contain_sensitive_or_escape_material():
    forbidden = (
        "token",
        "receipt",
        "authorization",
        "approval",
        "signed url",
        "http://",
        "https://",
        "/home/",
        "terminal",
        "shell",
        "python",
        "curl",
        "ffmpeg",
        "normalize.py",
        "hangzhou",
        "travel",
        "vlog",
    )
    for handler in tools.HANDLERS.values():
        overview = "\n".join(
            _natural_language_values(_parsed(handler, {"help": True}))
        ).lower()
        examples = "\n".join(
            _natural_language_values(
                _parsed(handler, {"help": True, "help_topic": "examples"})
            )
        ).lower()
        assert not any(value in overview for value in forbidden)
        assert not any(value in examples for value in forbidden)


def test_false_missing_and_string_help_follow_optional_task_identity_contract():
    name = "video_edit_preferences_resolve"
    assert schemas.validate_tool_arguments(name, {}) == []
    assert schemas.validate_tool_arguments(name, {"help": False}) == []

    string_result = _parsed(tools.HANDLERS[name], {"help": "true"})
    assert string_result["code"] == "invalid_arguments"
    assert any(
        issue["path"] == "$.help" and issue["rule"] == "type"
        for issue in string_result["issues"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"help": True, "action": "set"},
        {"help": True, "scope": "scene"},
        {"help": True, "scope": "scene", "action": "set"},
    ],
)
def test_help_true_ignores_business_conditionals_but_keeps_schema_boundaries(
    arguments,
):
    name = "video_edit_preferences_update"

    assert schemas.validate_tool_arguments(name, arguments) == []
    invalid_help = schemas.validate_tool_arguments(
        name, {"help": "true", "action": "set"}
    )
    assert any(
        issue["path"] == "$.help" and issue["rule"] == "type"
        for issue in invalid_help
    )
    assert schemas.validate_tool_arguments(
        name,
        {"help": True, "unknown_sensitive_key": "ignored?"},
    ) == [
        {
            "path": "$.unknown_sensitive_key",
            "rule": "additionalProperties",
        }
    ]


def test_direct_visible_help_then_business_uses_same_handler_once(monkeypatch):
    import model_tools

    name = "video_edit_preferences_resolve"
    handler_calls: list[str] = []
    boundary_calls = _install_resolve_boundary_trace(monkeypatch)
    _register_resolve_handler()
    trajectory: list[str] = []

    visible = next(
        definition["function"]
        for definition in model_tools.get_tool_definitions(
            enabled_toolsets=["video_edit"],
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        if definition["function"]["name"] == name
    )
    assert "Normal-call required inputs: none" in visible["parameters"][
        "description"
    ]
    assert "Required for normal calls" not in visible["parameters"]["properties"][
        "task_id"
    ]["description"]
    assert "allOf" not in visible["parameters"]
    assert visible["parameters"]["else"] == {"required": []}

    def observe_dispatch(
        dispatched_name: str,
        arguments: dict[str, Any],
        dispatch,
    ) -> str:
        assert dispatched_name == name
        mode = "help" if arguments.get("help") is True else "business"
        handler_calls.append(mode)
        return dispatch()

    def invoke(arguments: dict[str, Any]) -> dict[str, Any]:
        trajectory.append(name)
        return json.loads(
            model_tools.handle_function_call(
                name,
                arguments,
                enabled_toolsets=["video_edit"],
                dispatch_wrapper=observe_dispatch,
            )
        )

    help_result = invoke({"help": True})
    assert help_result["tool"] == name
    assert help_result["side_effects"] == "none"
    assert handler_calls == ["help"]
    assert boundary_calls == []

    business_result = invoke({"task_id": "TRACE_TASK_ID"})
    assert business_result["ok"] is True
    assert business_result["workflow_id"] == "trace-workflow-id"
    assert trajectory == [name, name]
    assert handler_calls == ["help", "business"]
    assert boundary_calls == [
        "agent_context",
        "state.workflow_id",
        "preferences.resolve",
        "state.create_or_validate_identity",
    ]


def test_deferred_exact_describe_help_business_trajectory(monkeypatch):
    import model_tools
    from tools import tool_search as tool_search_module

    name = "video_edit_preferences_resolve"
    handler_calls: list[str] = []
    boundary_calls = _install_resolve_boundary_trace(monkeypatch)
    _register_resolve_handler()
    trajectory: list[str] = []

    def unexpected_search(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("deferred help trajectory must not call tool_search")

    monkeypatch.setattr(tool_search_module, "dispatch_tool_search", unexpected_search)

    def observe_dispatch(
        dispatched_name: str,
        arguments: dict[str, Any],
        dispatch,
    ) -> str:
        assert dispatched_name == name
        mode = "help" if arguments.get("help") is True else "business"
        handler_calls.append(mode)
        return dispatch()

    def invoke(
        call_name: str,
        arguments: dict[str, Any],
        **call_kwargs: Any,
    ) -> dict[str, Any]:
        trajectory.append(call_name)
        return json.loads(
            model_tools.handle_function_call(
                call_name,
                arguments,
                enabled_toolsets=["video_edit"],
                dispatch_wrapper=observe_dispatch,
                **call_kwargs,
            )
        )

    described = invoke("tool_describe", {"name": name})
    assert described["name"] == name
    assert described["parameters"]["properties"]["help"]["type"] == "boolean"
    assert described["parameters"]["else"] == {"required": []}
    assert "Normal-call required inputs: none" in described["parameters"][
        "description"
    ]
    assert "Required for normal calls" not in described["parameters"]["properties"][
        "task_id"
    ]["description"]

    missing_result = invoke(
        "tool_call",
        {"name": name, "arguments": {}},
        turn_id="TRACE_TASK_ID",
    )
    assert missing_result["ok"] is True
    assert missing_result["workflow_id"] == "trace-workflow-id"
    assert handler_calls == ["business"]
    assert boundary_calls == [
        "agent_context",
        "state.workflow_id",
        "preferences.resolve",
        "state.create_or_validate_identity",
    ]

    help_result = invoke(
        "tool_call",
        {"name": name, "arguments": {"help": True, "scene": "SCENE_CATEGORY"}},
    )
    assert help_result["tool"] == name
    assert help_result["ignored_business_fields"] == ["scene"]
    assert help_result["side_effects"] == "none"
    assert handler_calls == ["business", "help"]
    assert len(boundary_calls) == 4

    business_result = invoke(
        "tool_call",
        {"name": name, "arguments": {"task_id": "TRACE_TASK_ID"}},
    )
    assert business_result["ok"] is True
    assert trajectory == ["tool_describe", "tool_call", "tool_call", "tool_call"]
    assert "tool_search" not in trajectory
    assert handler_calls == ["business", "help", "business"]
    assert boundary_calls == [
        "agent_context",
        "state.workflow_id",
        "preferences.resolve",
        "state.create_or_validate_identity",
    ] * 2
