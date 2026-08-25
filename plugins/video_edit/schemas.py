"""Schemas and pure self-description for the video_edit atomic tools."""

from __future__ import annotations

import copy
import math
from typing import Any

from plugins.video_edit import preferences
from plugins.video_edit.paths import MAX_TASK_ID_LENGTH


HELP_SCHEMA_VERSION = "1.6"
HELP_TOPICS = ("overview", "inputs", "outputs", "errors", "recovery", "examples")
HELP_CONTROL_FIELDS = frozenset({"help", "help_topic"})
_SELF_TOOL = "$self"

_HELP_PROPERTIES = {
    "help": {
        "type": "boolean",
        "default": False,
        "description": "Return this tool's side-effect-free contract instead of executing it.",
    },
    "help_topic": {
        "type": "string",
        "enum": list(HELP_TOPICS),
        "default": "overview",
        "description": "Select one section of the side-effect-free help contract.",
    },
}

_FORBIDDEN_FALLBACKS = (
    "Do not replace this plugin call with a general execution tool.",
    "Do not make provider requests outside the plugin.",
    "Do not create parallel workflow state or change stable identifiers during retry.",
    "Do not repeat a completed stage when its result can be reused.",
)

_COMMON_INVALID_ARGUMENT_ERROR = {
    "code": "invalid_arguments",
    "reason_code": "invalid_arguments",
    "retryable": True,
    "next_tool": _SELF_TOOL,
    "recovery": "Correct the reported fields and call the same tool once.",
}

_INVALID_HELP_TOPIC_ERROR = {
    "code": "invalid_arguments",
    "reason_code": "invalid_help_topic",
    "retryable": True,
    "next_tool": _SELF_TOOL,
    "recovery": "Choose one allowed help_topic and call Help once more.",
}

_SERVICE_ADMISSION_ERROR = {
    "reason_code": "service_admission_failed",
    "retryable": False,
    "next_tool": None,
    "recovery": "Stop. Surface the video service admission failure without retrying or bypassing the plugin.",
}

_SERVICE_REQUEST_REJECTED_ERROR = {
    "reason_code": "service_request_rejected",
    "retryable": False,
    "next_tool": None,
    "recovery": "Stop. Surface that the video service rejected this request without retrying or bypassing the plugin.",
}

_LOCAL_STATE_UNAVAILABLE_ERROR = {
    "reason_code": "local_state_unavailable",
    "retryable": False,
    "next_tool": None,
    "recovery": "Stop. Preserve the request until local video state is available again.",
}

_WORKFLOW_UNAVAILABLE_ERROR = {
    "reason_code": "workflow_unavailable",
    "retryable": False,
    "next_tool": None,
    "recovery": "Stop. Surface the unavailable trusted workflow checkpoint without retrying or bypassing the plugin.",
}

TOOL_HELP_METADATA: dict[str, dict[str, Any]] = {}


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool": name, "arguments": arguments}


def _help(
    *,
    when_to_use: str,
    cross_field_invariants: list[str],
    reusable_business_ids: list[str],
    success_outputs: list[str],
    failure_code: str,
    failure_recovery: str,
    next_tools: list[dict[str, Any]],
    minimal_valid_call: dict[str, Any],
    common_mistake: dict[str, Any],
    corrected_call: dict[str, Any],
    bad_recovery: str,
    recoverable_errors: list[dict[str, Any]] | None = None,
    terminal_errors: list[dict[str, Any]] | None = None,
    service_call: bool = False,
    workflow_required: bool = False,
) -> dict[str, Any]:
    recoverable = [
        dict(_COMMON_INVALID_ARGUMENT_ERROR),
        dict(_INVALID_HELP_TOPIC_ERROR),
        {
            "code": failure_code,
            "reason_code": "transient_failure",
            "retryable": True,
            "next_tool": _SELF_TOOL,
            "recovery": failure_recovery,
        },
    ]
    recoverable.extend(copy.deepcopy(recoverable_errors or []))
    terminal: list[dict[str, Any]] = [
        {"code": failure_code, **_LOCAL_STATE_UNAVAILABLE_ERROR}
    ]
    if service_call:
        terminal.append({"code": failure_code, **_SERVICE_ADMISSION_ERROR})
        terminal.append({"code": failure_code, **_SERVICE_REQUEST_REJECTED_ERROR})
    if workflow_required:
        terminal.append({"code": failure_code, **_WORKFLOW_UNAVAILABLE_ERROR})
    terminal.extend(copy.deepcopy(terminal_errors or []))
    for item in recoverable + terminal:
        item.setdefault("code", failure_code)
        if any(
            key not in item
            for key in ("reason_code", "retryable", "next_tool", "recovery")
        ):
            raise ValueError("video help error metadata is incomplete")
    return {
        "when_to_use": when_to_use,
        "cross_field_invariants": cross_field_invariants,
        "reusable_business_ids": reusable_business_ids,
        "success_outputs": success_outputs,
        "recoverable_errors": recoverable,
        "terminal_errors": terminal,
        "next_tools": next_tools,
        "minimal_valid_call": minimal_valid_call,
        "common_mistake": common_mistake,
        "corrected_call": corrected_call,
        "bad_recovery": bad_recovery,
        "forbidden_fallbacks": list(_FORBIDDEN_FALLBACKS),
    }


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    help_metadata: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    if HELP_CONTROL_FIELDS.intersection(properties):
        raise ValueError(f"{name} business properties overlap help controls")
    business_required = list(required or [])
    visible_properties = copy.deepcopy(properties)
    for field in business_required:
        field_schema = visible_properties[field]
        current = str(field_schema.get("description") or "").strip()
        required_note = "Required for normal calls; omit only when help=true."
        field_schema["description"] = f"{current} {required_note}".strip()
    bound_metadata = copy.deepcopy(help_metadata)
    error_transitions: list[dict[str, Any]] = []
    for item in bound_metadata["recoverable_errors"] + bound_metadata["terminal_errors"]:
        if item["next_tool"] == _SELF_TOOL:
            item["next_tool"] = name
        error_transitions.append(
            {"when": f"error:{item['reason_code']}", "tool": item["next_tool"]}
        )
    bound_metadata["next_tools"] = list(bound_metadata["next_tools"]) + error_transitions
    TOOL_HELP_METADATA[name] = bound_metadata
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "description": (
                "Normal-call required inputs: "
                f"{', '.join(business_required) or 'none'}. A literal help=true "
                "call needs no business inputs and ignores supplied business fields."
            ),
            "properties": visible_properties | copy.deepcopy(_HELP_PROPERTIES),
            "required": [],
            "if": {
                "properties": {"help": {"const": True}},
                "required": ["help"],
            },
            "then": {},
            "else": {"required": business_required},
            "additionalProperties": False,
        },
        "emoji": "🎬",
    }


_PREFERENCES = {
    "type": "object",
    "description": "Creative choices. Omitted fields are selected from remembered preferences or sensible defaults.",
    "properties": {
        "style": {"type": "string", "maxLength": preferences.MAX_STYLE_LENGTH},
        "aspect_ratio": {"type": "string", "enum": ["9:16", "16:9", "1:1"]},
        "duration": {"type": "integer", "minimum": 1, "maximum": 3600},
        "decision_mode": {"type": "string", "enum": ["auto", "balanced", "precise"]},
        "editing_directives": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(preferences.VALID_DIRECTIVES)},
            "maxItems": preferences.MAX_EDITING_DIRECTIVES,
        },
        "upload_preference": {"type": "string", "enum": ["raw_direct", "normalized"]},
        "user_prompt": {
            "type": "string",
            "maxLength": preferences.MAX_USER_PROMPT_LENGTH,
        },
    },
}


TOOL_DEFINITIONS = [
    _tool(
        "video_edit_preferences_resolve",
        "Resolve explicit, remembered, and default creative choices for one video-edit task and continue with sensible defaults.",
        {
            "task_id": {
                "type": "string",
                "maxLength": MAX_TASK_ID_LENGTH,
                "description": "Stable key for one edit: reuse it for retry/resume, choose a new value for an explicit re-edit.",
            },
            "scene": {"type": "string", "maxLength": 64, "default": "general"},
            "preferences": _PREFERENCES,
            "silent": {
                "type": "boolean",
                "default": False,
                "description": "True for preference-hit or proactive runs; unresolved choices use defaults.",
            },
        },
        _help(
            when_to_use="Start a new interactive edit or resolve choices for a stable task before upload.",
            cross_field_invariants=[
                "Reuse task_id for resume or retry; use a new task_id only for an explicit re-edit.",
                "Explicit preferences override remembered values and defaults.",
            ],
            reusable_business_ids=["task_id"],
            success_outputs=["ok", "workflow_id", "scene", "preferences", "sources", "memory_hit", "next"],
            failure_code="preferences_resolve_failed",
            failure_recovery="Retry once with the same task_id after correcting the reported condition.",
            workflow_required=True,
            next_tools=[{"when": "success", "tool": "video_edit_upload_assets"}],
            minimal_valid_call=_call("video_edit_preferences_resolve", {"task_id": "TASK_ID"}),
            common_mistake=_call("video_edit_preferences_resolve", {"scene": "SCENE_CATEGORY"}),
            corrected_call=_call("video_edit_preferences_resolve", {"task_id": "TASK_ID"}),
            bad_recovery="Changing task_id during a retry creates a different edit and loses the continuation boundary.",
        ),
        ["task_id"],
    ),
    _tool(
        "video_edit_preferences_update",
        "Set or forget a bounded hard or soft video preference for the current agent.",
        {
            "scope": {"type": "string", "enum": ["global", "scene"]},
            "scene": {"type": "string", "maxLength": 64, "default": "general"},
            "kind": {"type": "string", "enum": ["hard", "soft"]},
            "action": {"type": "string", "enum": ["set", "forget"]},
            "preferences": _PREFERENCES,
        },
        _help(
            when_to_use="Use only when the user explicitly asks to remember, change, or forget video preferences.",
            cross_field_invariants=[
                "scope=scene applies only to the exact semantic scene category supplied; scope=global applies unconditionally.",
                "The schema cannot encode arbitrary predicates. Never drop qualifiers, reinterpret a qualifier as scene, or broaden a conditional preference to fit global or scene; do not call this tool when any qualifier cannot be represented.",
                "action=set applies supplied preferences; action=forget removes supplied fields or all fields when omitted.",
            ],
            reusable_business_ids=[],
            success_outputs=["ok", "scope", "kind", "action", "scene"],
            failure_code="preferences_update_failed",
            failure_recovery="Correct the preference fields and retry the same update once.",
            next_tools=[{"when": "success", "tool": None}],
            minimal_valid_call=_call(
                "video_edit_preferences_update",
                {"scope": "global", "kind": "soft", "action": "forget"},
            ),
            common_mistake=_call(
                "video_edit_preferences_update",
                {"scope": "scene", "kind": "soft", "action": "set", "preferences": {"decision_mode": "auto"}},
            ),
            corrected_call=_call(
                "video_edit_preferences_update",
                {
                    "scope": "scene",
                    "scene": "SCENE_CATEGORY",
                    "kind": "soft",
                    "action": "set",
                    "preferences": {"decision_mode": "auto"},
                },
            ),
            bad_recovery="Do not turn preference maintenance into a separate edit workflow.",
        ),
        ["scope", "kind", "action"],
    ),
    _tool(
        "video_edit_preferences_record_success",
        "Record only confirmed creative choices after a successful rendered delivery so later edits can be silent.",
        {
            "scene": {"type": "string", "maxLength": 64, "default": "general"},
            "preferences": _PREFERENCES,
            "confirmed_fields": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
                "default": [],
            },
        },
        _help(
            when_to_use="Optionally record confirmed choices only after an interactive result was downloaded successfully.",
            cross_field_invariants=[
                "Call only after successful delivery.",
                "confirmed_fields limits which supplied preferences are recorded; an empty list uses all recordable supplied fields.",
            ],
            reusable_business_ids=[],
            success_outputs=["ok", "recorded", "scope", "kind", "action", "scene"],
            failure_code="preferences_record_failed",
            failure_recovery="Retry once with the same confirmed preference values.",
            next_tools=[{"when": "success", "tool": None}],
            minimal_valid_call=_call(
                "video_edit_preferences_record_success",
                {"preferences": {"decision_mode": "auto"}},
            ),
            common_mistake=_call("video_edit_preferences_record_success", {}),
            corrected_call=_call(
                "video_edit_preferences_record_success",
                {"preferences": {"decision_mode": "auto"}},
            ),
            bad_recovery="Do not record unconfirmed choices or call this before a result is delivered.",
        ),
        ["preferences"],
    ),
    _tool(
        "video_edit_upload_assets",
        "Upload bounded video assets for a workflow. Interactive edits pass files; proactive edits omit files and reuse the plugin-owned manifest checkpoint. Paths are validated locally.",
        {
            "workflow_id": {"type": "string"},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "normalize": {
                "type": "boolean",
                "description": "When omitted, the plugin chooses direct upload or its installed media adapter.",
            },
        },
        _help(
            when_to_use="Upload the selected interactive media or continue a proactive workflow after preferences are resolved.",
            cross_field_invariants=[
                "Interactive workflows supply one to eight trusted media references; proactive workflows reuse their stored manifest selection.",
                "Reuse workflow_id and the exact source selection for retry; a changed selection is a new edit.",
            ],
            reusable_business_ids=["workflow_id"],
            success_outputs=["ok", "workflow_id", "uploaded", "reused", "strategy", "next"],
            failure_code="upload_assets_failed",
            failure_recovery="Retry with the same workflow_id and exact media selection so accepted uploads are reused.",
            recoverable_errors=[
                {
                    "reason_code": "media_preparation_failed",
                    "retryable": True,
                    "next_tool": "video_edit_upload_assets",
                    "recovery": "Retry the same upload once; keep the workflow and media selection unchanged.",
                }
            ],
            service_call=True,
            workflow_required=True,
            next_tools=[{"when": "success_or_reused", "tool": "video_edit_create_project"}],
            minimal_valid_call=_call("video_edit_upload_assets", {"workflow_id": "WORKFLOW_ID_FROM_PREVIOUS_TOOL"}),
            common_mistake=_call("video_edit_upload_assets", {"files": ["MEDIA_REFERENCE_FROM_USER"]}),
            corrected_call=_call(
                "video_edit_upload_assets",
                {
                    "workflow_id": "WORKFLOW_ID_FROM_PREVIOUS_TOOL",
                    "files": ["MEDIA_REFERENCE_FROM_USER"],
                },
            ),
            bad_recovery="Do not change workflow_id, replace the source selection, or repeat already accepted uploads.",
        ),
        ["workflow_id"],
    ),
    _tool(
        "video_edit_create_project",
        "Create or resume one rendered cloud-edit project from uploaded workflow assets.",
        {
            "workflow_id": {"type": "string"},
            "user_prompt": {"type": "string", "maxLength": 512, "default": ""},
        },
        _help(
            when_to_use="Create the render project after every source in the same workflow has uploaded successfully.",
            cross_field_invariants=[
                "All selected assets must be uploaded before project creation.",
                "Reuse workflow_id so an existing project is returned instead of duplicated.",
            ],
            reusable_business_ids=["workflow_id"],
            success_outputs=["ok", "workflow_id", "project_id", "reused", "next"],
            failure_code="create_project_failed",
            failure_recovery="Retry with the same workflow_id; an existing project is reused.",
            recoverable_errors=[
                {
                    "reason_code": "upload_incomplete",
                    "retryable": True,
                    "next_tool": "video_edit_upload_assets",
                    "recovery": "Return to upload with the same workflow_id and source selection.",
                }
            ],
            service_call=True,
            workflow_required=True,
            next_tools=[{"when": "success_or_reused", "tool": "video_edit_wait_project"}],
            minimal_valid_call=_call("video_edit_create_project", {"workflow_id": "WORKFLOW_ID_FROM_UPLOAD"}),
            common_mistake=_call("video_edit_create_project", {"user_prompt": "COMPLETE_USER_INTENT"}),
            corrected_call=_call(
                "video_edit_create_project",
                {"workflow_id": "WORKFLOW_ID_FROM_UPLOAD", "user_prompt": "COMPLETE_USER_INTENT"},
            ),
            bad_recovery="Do not create a second workflow or project when the existing workflow can be resumed.",
        ),
        ["workflow_id"],
    ),
    _tool(
        "video_edit_wait_project",
        "Poll a rendered project for a bounded interval. Repeat with the same workflow when continue_required is true.",
        {
            "workflow_id": {"type": "string"},
            "max_wait_seconds": {"type": "integer", "minimum": 15, "maximum": 480, "default": 120},
        },
        _help(
            when_to_use="Poll the project after creation and continue with the same workflow while processing remains incomplete.",
            cross_field_invariants=[
                "The workflow must already contain a created project.",
                "When continue_required is true, call this tool again with the same workflow_id.",
            ],
            reusable_business_ids=["workflow_id"],
            success_outputs=["ok", "workflow_id", "project_id", "status", "continue_required", "next"],
            failure_code="wait_project_failed",
            failure_recovery="Retry the bounded poll with the same workflow_id.",
            recoverable_errors=[
                {
                    "reason_code": "upload_incomplete",
                    "retryable": True,
                    "next_tool": "video_edit_upload_assets",
                    "recovery": "Finish uploading the same workflow before creating or waiting for its project.",
                },
                {
                    "reason_code": "project_not_created",
                    "retryable": True,
                    "next_tool": "video_edit_create_project",
                    "recovery": "Create the project once from the completed upload checkpoint.",
                },
                {
                    "reason_code": "project_not_completed",
                    "retryable": True,
                    "next_tool": "video_edit_wait_project",
                    "recovery": "Continue polling the existing project with the same workflow_id.",
                },
            ],
            terminal_errors=[
                {
                    "reason_code": "project_terminal",
                    "retryable": False,
                    "next_tool": None,
                    "recovery": "Stop when the project status is failed, cancelled, or error.",
                }
            ],
            service_call=True,
            workflow_required=True,
            next_tools=[
                {"when": "processing", "tool": "video_edit_wait_project"},
                {"when": "completed", "tool": "video_edit_download_result"},
                {"when": "failed_cancelled_or_error", "tool": None},
            ],
            minimal_valid_call=_call("video_edit_wait_project", {"workflow_id": "WORKFLOW_ID_FROM_PROJECT"}),
            common_mistake=_call("video_edit_wait_project", {"project_id": "PROJECT_ID_FROM_PROJECT"}),
            corrected_call=_call("video_edit_wait_project", {"workflow_id": "WORKFLOW_ID_FROM_PROJECT"}),
            bad_recovery="Do not create another project while the current project is still processing.",
        ),
        ["workflow_id"],
    ),
    _tool(
        "video_edit_download_result",
        "Download a completed rendered result into the active agent output directory and persist the resume checkpoint.",
        {
            "workflow_id": {"type": "string"},
            "filename": {"type": "string", "maxLength": 128},
        },
        _help(
            when_to_use="Download only after the project reached completed status, or resume the same pending download.",
            cross_field_invariants=[
                "The workflow must contain a completed project result.",
                "Reuse workflow_id so an existing or partially committed result is recovered rather than downloaded twice.",
            ],
            reusable_business_ids=["workflow_id"],
            success_outputs=["ok", "workflow_id", "output", "size", "sha256", "reused", "recovered", "next"],
            failure_code="download_result_failed",
            failure_recovery="Retry with the same workflow_id so the saved download checkpoint is reused.",
            recoverable_errors=[
                {
                    "reason_code": "upload_incomplete",
                    "retryable": True,
                    "next_tool": "video_edit_upload_assets",
                    "recovery": "Finish uploading the same workflow before continuing.",
                },
                {
                    "reason_code": "project_not_created",
                    "retryable": True,
                    "next_tool": "video_edit_create_project",
                    "recovery": "Create the project from the completed upload checkpoint.",
                },
                {
                    "reason_code": "project_not_completed",
                    "retryable": True,
                    "next_tool": "video_edit_wait_project",
                    "recovery": "Wait for the existing project with the same workflow_id.",
                },
            ],
            terminal_errors=[
                {
                    "reason_code": "project_terminal",
                    "retryable": False,
                    "next_tool": None,
                    "recovery": "Stop when the project status is failed, cancelled, or error.",
                }
            ],
            service_call=True,
            workflow_required=True,
            next_tools=[
                {"when": "interactive_success", "tool": "video_edit_preferences_record_success"},
                {"when": "proactive_success", "tool": "video_edit_proactive_report"},
            ],
            minimal_valid_call=_call("video_edit_download_result", {"workflow_id": "WORKFLOW_ID_FROM_WAIT"}),
            common_mistake=_call("video_edit_download_result", {"filename": "OUTPUT_FILENAME"}),
            corrected_call=_call("video_edit_download_result", {"workflow_id": "WORKFLOW_ID_FROM_WAIT"}),
            bad_recovery="Do not change workflow_id or bypass the pending download checkpoint.",
        ),
        ["workflow_id"],
    ),
    _tool(
        "video_edit_proactive_resolve",
        "Resolve the server-owned weekly-memory manifest into a private workflow without exposing source paths to the model.",
        {
            "manifest_id": {"type": "string"},
            "task_id": {"type": "string", "maxLength": MAX_TASK_ID_LENGTH},
        },
        _help(
            when_to_use="Start a proactive weekly edit only from trusted context that supplies both identifiers.",
            cross_field_invariants=[
                "Both manifest_id and task_id must come from trusted proactive context; ordinary user text is insufficient.",
                "Reuse both identifiers for retry so the same private workflow is recovered.",
            ],
            reusable_business_ids=["manifest_id", "task_id"],
            success_outputs=["ok", "workflow_id", "manifest_id", "file_count", "silent", "next"],
            failure_code="proactive_resolve_failed",
            failure_recovery="Retry once with the same trusted manifest_id and task_id.",
            service_call=True,
            workflow_required=True,
            next_tools=[{"when": "success", "tool": "video_edit_upload_assets"}],
            minimal_valid_call=_call(
                "video_edit_proactive_resolve",
                {"manifest_id": "MANIFEST_ID_FROM_TRUSTED_CONTEXT", "task_id": "TASK_ID_FROM_TRUSTED_CONTEXT"},
            ),
            common_mistake=_call(
                "video_edit_proactive_resolve",
                {"manifest_id": "MANIFEST_ID_FROM_TRUSTED_CONTEXT"},
            ),
            corrected_call=_call(
                "video_edit_proactive_resolve",
                {"manifest_id": "MANIFEST_ID_FROM_TRUSTED_CONTEXT", "task_id": "TASK_ID_FROM_TRUSTED_CONTEXT"},
            ),
            bad_recovery="Do not infer either identifier from ordinary text or substitute a new task_id during retry.",
        ),
        ["manifest_id", "task_id"],
    ),
    _tool(
        "video_edit_proactive_report",
        "Report a successfully downloaded weekly-memory result to local-server exactly once.",
        {"workflow_id": {"type": "string"}},
        _help(
            when_to_use="Report a proactive workflow only after its result was downloaded successfully.",
            cross_field_invariants=[
                "The workflow must be proactive and have a delivered output.",
                "Reuse workflow_id; a completed report returns the idempotent result without reporting twice.",
            ],
            reusable_business_ids=["workflow_id"],
            success_outputs=["ok", "workflow_id", "reported", "reused", "result"],
            failure_code="proactive_report_failed",
            failure_recovery="Retry with the same workflow_id so an already completed report is reused.",
            recoverable_errors=[
                {
                    "reason_code": "upload_incomplete",
                    "retryable": True,
                    "next_tool": "video_edit_upload_assets",
                    "recovery": "Finish uploading the same proactive workflow before continuing.",
                },
                {
                    "reason_code": "project_not_created",
                    "retryable": True,
                    "next_tool": "video_edit_create_project",
                    "recovery": "Create the project from the completed upload checkpoint.",
                },
                {
                    "reason_code": "project_not_completed",
                    "retryable": True,
                    "next_tool": "video_edit_wait_project",
                    "recovery": "Wait for the existing project with the same workflow_id.",
                },
                {
                    "reason_code": "result_not_downloaded",
                    "retryable": True,
                    "next_tool": "video_edit_download_result",
                    "recovery": "Download and validate the completed result before reporting it.",
                },
            ],
            terminal_errors=[
                {
                    "reason_code": "not_proactive_workflow",
                    "retryable": False,
                    "next_tool": None,
                    "recovery": "Stop because an interactive workflow cannot be reported as a proactive task.",
                },
                {
                    "reason_code": "project_terminal",
                    "retryable": False,
                    "next_tool": None,
                    "recovery": "Stop when the project status is failed, cancelled, or error.",
                },
            ],
            service_call=True,
            workflow_required=True,
            next_tools=[{"when": "success_or_reused", "tool": None}],
            minimal_valid_call=_call("video_edit_proactive_report", {"workflow_id": "WORKFLOW_ID_FROM_DOWNLOAD"}),
            common_mistake=_call(
                "video_edit_proactive_report",
                {"manifest_id": "MANIFEST_ID_FROM_TRUSTED_CONTEXT"},
            ),
            corrected_call=_call("video_edit_proactive_report", {"workflow_id": "WORKFLOW_ID_FROM_DOWNLOAD"}),
            bad_recovery="Do not report before download or create a second workflow to repeat the report.",
        ),
        ["workflow_id"],
    ),
]

TOOL_DEFINITIONS_BY_NAME = {definition["name"]: definition for definition in TOOL_DEFINITIONS}


def _walk_contract(
    schema: dict[str, Any],
    path: str,
    defaults: dict[str, Any],
    constraints: dict[str, Any],
) -> None:
    if "default" in schema:
        defaults[path] = copy.deepcopy(schema["default"])
    selected = {
        key: copy.deepcopy(schema[key])
        for key in ("enum", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems")
        if key in schema
    }
    if selected:
        constraints[path] = selected
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if isinstance(child, dict):
                _walk_contract(child, f"{path}.{name}" if path else name, defaults, constraints)
    items = schema.get("items")
    if isinstance(items, dict):
        _walk_contract(items, f"{path}[]", defaults, constraints)


def business_required_names(name: str) -> list[str]:
    """Read normal-mode required fields from the public conditional schema."""
    parameters = TOOL_DEFINITIONS_BY_NAME[name]["parameters"]
    condition = parameters.get("if")
    otherwise = parameters.get("else")
    if (
        isinstance(condition, dict)
        and condition.get("required") == ["help"]
        and isinstance(condition.get("properties"), dict)
        and condition["properties"].get("help") == {"const": True}
        and parameters.get("then") == {}
        and isinstance(otherwise, dict)
        and isinstance(otherwise.get("required"), list)
    ):
        return list(otherwise["required"])

    # Read the pre-1.5 wrapper as a compatibility fallback for persisted or
    # third-party copies of the schema-derived Help contract.
    for clause in parameters.get("allOf") or []:
        if not isinstance(clause, dict):
            continue
        condition = clause.get("if")
        otherwise = clause.get("else")
        if (
            isinstance(condition, dict)
            and condition.get("required") == ["help"]
            and isinstance(condition.get("properties"), dict)
            and condition["properties"].get("help") == {"const": True}
            and isinstance(otherwise, dict)
            and isinstance(otherwise.get("required"), list)
        ):
            return list(otherwise["required"])
    return list(parameters.get("required") or [])


def technical_contract(name: str) -> dict[str, Any]:
    """Return schema-derived input facts used by Help and its validator tests."""
    definition = TOOL_DEFINITIONS_BY_NAME[name]
    parameters = definition["parameters"]
    properties = parameters["properties"]
    required_names = business_required_names(name)
    required = {key: copy.deepcopy(properties[key]) for key in required_names}
    optional = {
        key: copy.deepcopy(value)
        for key, value in properties.items()
        if key not in required_names
    }
    defaults: dict[str, Any] = {}
    constraints: dict[str, Any] = {}
    for key, value in properties.items():
        _walk_contract(value, key, defaults, constraints)
    return {
        "required_inputs": required,
        "optional_inputs": optional,
        "defaults": defaults,
        "enums_and_ranges": constraints,
    }


def business_input_names(name: str) -> frozenset[str]:
    properties = TOOL_DEFINITIONS_BY_NAME[name]["parameters"]["properties"]
    return frozenset(properties).difference(HELP_CONTROL_FIELDS)


def business_failure_code(name: str) -> str:
    errors = TOOL_HELP_METADATA[name]["recoverable_errors"]
    return next(item["code"] for item in errors if item["reason_code"] == "transient_failure")


def error_contract(name: str, reason_code: str) -> dict[str, Any]:
    """Return the Help error contract consumed by runtime error responses."""
    metadata = TOOL_HELP_METADATA[name]
    for item in metadata["recoverable_errors"] + metadata["terminal_errors"]:
        if item["reason_code"] == reason_code:
            return copy.deepcopy(item)
    raise KeyError(f"unknown video error reason: {name}:{reason_code}")


def terminal_failure_code(name: str) -> str:
    return TOOL_HELP_METADATA[name]["terminal_errors"][0]["code"]


def render_tool_help(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Render a deterministic help response without reading runtime state."""
    definition = TOOL_DEFINITIONS_BY_NAME[name]
    metadata = TOOL_HELP_METADATA[name]
    known_business_fields = business_input_names(name)
    ignored = sorted(key for key in args if key in known_business_fields)
    raw_topic = args.get("help_topic", "overview")
    topic = raw_topic if isinstance(raw_topic, str) else ""
    base = {
        "tool": name,
        "schema_version": HELP_SCHEMA_VERSION,
        "help_topic": topic if topic in HELP_TOPICS else "invalid",
        "side_effects": "none",
        "ignored_business_fields": ignored,
    }
    if topic not in HELP_TOPICS:
        contract = error_contract(name, "invalid_help_topic")
        return base | {
            "error": "help_topic is not supported",
            "code": contract["code"],
            "reason_code": contract["reason_code"],
            "retryable": contract["retryable"],
            "next": contract["next_tool"],
            "recovery": contract["recovery"],
            "allowed_topics": list(HELP_TOPICS),
        }

    technical = technical_contract(name)
    retryable = {
        item["reason_code"]: bool(item["retryable"])
        for item in metadata["recoverable_errors"] + metadata["terminal_errors"]
    }
    overview = base | {
        "purpose": definition["description"],
        "when_to_use": metadata["when_to_use"],
        **technical,
        "cross_field_invariants": copy.deepcopy(metadata["cross_field_invariants"]),
        "reusable_business_ids": list(metadata["reusable_business_ids"]),
        "success_outputs": list(metadata["success_outputs"]),
        "recoverable_errors": copy.deepcopy(metadata["recoverable_errors"]),
        "terminal_errors": copy.deepcopy(metadata["terminal_errors"]),
        "retryable": retryable,
        "next_tools": copy.deepcopy(metadata["next_tools"]),
        "minimal_valid_call": copy.deepcopy(metadata["minimal_valid_call"]),
        "common_mistake": copy.deepcopy(metadata["common_mistake"]),
        "corrected_call": copy.deepcopy(metadata["corrected_call"]),
        "bad_recovery": metadata["bad_recovery"],
        "forbidden_fallbacks": list(metadata["forbidden_fallbacks"]),
    }
    if topic == "overview":
        return overview

    topic_fields = {
        "inputs": (
            "purpose",
            "when_to_use",
            "required_inputs",
            "optional_inputs",
            "defaults",
            "enums_and_ranges",
            "cross_field_invariants",
            "reusable_business_ids",
            "minimal_valid_call",
        ),
        "outputs": ("purpose", "success_outputs", "next_tools"),
        "errors": ("recoverable_errors", "terminal_errors", "retryable"),
        "recovery": (
            "recoverable_errors",
            "terminal_errors",
            "retryable",
            "next_tools",
            "common_mistake",
            "corrected_call",
            "bad_recovery",
            "forbidden_fallbacks",
        ),
    }
    if topic == "examples":
        examples = [metadata["minimal_valid_call"]]
        if metadata["corrected_call"] != metadata["minimal_valid_call"]:
            examples.append(metadata["corrected_call"])
        return base | {"examples": copy.deepcopy(examples)}
    return base | {key: copy.deepcopy(overview[key]) for key in topic_fields[topic]}


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if expected == "number":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, float) and math.isfinite(value)
    return True


def _validate_value(value: Any, schema: dict[str, Any], path: str, issues: list[dict[str, Any]]) -> None:
    if "const" in schema and value != schema["const"]:
        issues.append({"path": path, "rule": "const", "expected": copy.deepcopy(schema["const"])})
        return
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(value, expected):
        issues.append({"path": path, "rule": "type", "expected": expected})
        return
    if "enum" in schema and value not in schema["enum"]:
        issues.append({"path": path, "rule": "enum", "allowed": copy.deepcopy(schema["enum"])})
        return
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            issues.append({"path": path, "rule": "maxLength", "maximum": schema["maxLength"]})
    elif isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            issues.append({"path": path, "rule": "minItems", "minimum": schema["minItems"]})
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            issues.append({"path": path, "rule": "maxItems", "maximum": schema["maxItems"]})
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]", issues)
    elif isinstance(value, dict):
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for field in required:
            if isinstance(field, str) and field not in value:
                issues.append({"path": f"{path}.{field}", "rule": "required"})
        properties = schema.get("properties")
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False:
                for field in value:
                    if field not in properties:
                        issues.append({"path": f"{path}.{field}", "rule": "additionalProperties"})
            for field, child in properties.items():
                if field in value and isinstance(child, dict):
                    _validate_value(value[field], child, f"{path}.{field}", issues)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append({"path": path, "rule": "minimum", "minimum": schema["minimum"]})
        if "maximum" in schema and value > schema["maximum"]:
            issues.append({"path": path, "rule": "maximum", "maximum": schema["maximum"]})
    for subschema in schema.get("allOf") or []:
        if isinstance(subschema, dict):
            _validate_value(value, subschema, path, issues)
    condition = schema.get("if")
    if isinstance(condition, dict):
        condition_issues: list[dict[str, Any]] = []
        _validate_value(value, condition, path, condition_issues)
        branch = schema.get("then") if not condition_issues else schema.get("else")
        if isinstance(branch, dict):
            _validate_value(value, branch, path, issues)


def validate_tool_arguments(name: str, args: Any) -> list[dict[str, Any]]:
    """Validate the bounded JSON-schema subset used by video_edit tools."""
    if not isinstance(args, dict):
        return [{"path": "$", "rule": "type", "expected": "object"}]
    parameters = TOOL_DEFINITIONS_BY_NAME[name]["parameters"]
    issues: list[dict[str, Any]] = []
    _validate_value(args, parameters, "$", issues)
    return issues
