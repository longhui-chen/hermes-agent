"""JSON schemas for the video_edit plugin's atomic tools."""

from __future__ import annotations


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        "emoji": "🎬",
    }


_PREFERENCES = {
    "type": "object",
    "description": "Creative choices. Omitted fields are selected from remembered preferences or sensible defaults.",
    "properties": {
        "style": {"type": "string"},
        "aspect_ratio": {"type": "string", "enum": ["9:16", "16:9", "1:1"]},
        "duration": {"type": "integer", "minimum": 1, "maximum": 3600},
        "decision_mode": {"type": "string", "enum": ["auto", "balanced", "precise"]},
        "editing_directives": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "upload_preference": {"type": "string", "enum": ["raw_direct", "normalized"]},
        "user_prompt": {"type": "string", "maxLength": 512},
    },
}


TOOL_DEFINITIONS = [
    _tool(
        "video_edit_preferences_resolve",
        "Resolve explicit, remembered, and default creative choices for one video-edit task. This never asks the user for authorization.",
        {
            "task_id": {"type": "string", "description": "Stable task key for retry/resume."},
            "scene": {"type": "string", "maxLength": 64},
            "preferences": _PREFERENCES,
            "silent": {"type": "boolean", "description": "True for memory-hit or proactive runs; unresolved choices use defaults."},
        },
        ["task_id"],
    ),
    _tool(
        "video_edit_preferences_update",
        "Set or forget a bounded hard or soft video preference in the active profile.",
        {
            "scope": {"type": "string", "enum": ["global", "scene"]},
            "scene": {"type": "string", "maxLength": 64},
            "kind": {"type": "string", "enum": ["hard", "soft"]},
            "action": {"type": "string", "enum": ["set", "forget"]},
            "preferences": _PREFERENCES,
        },
        ["scope", "kind", "action"],
    ),
    _tool(
        "video_edit_preferences_record_success",
        "Record only confirmed creative choices after a successful rendered delivery so later edits can be silent.",
        {
            "scene": {"type": "string", "maxLength": 64},
            "preferences": _PREFERENCES,
            "confirmed_fields": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        ["preferences"],
    ),
    _tool(
        "video_edit_upload_assets",
        "Upload bounded video assets for a workflow. Interactive edits pass files; proactive edits omit files and reuse the plugin-owned manifest checkpoint. Paths are validated locally and never become an authorization token.",
        {
            "workflow_id": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "normalize": {"type": "boolean", "description": "Use the installed media normalizer when direct upload is unsuitable."},
        },
        ["workflow_id"],
    ),
    _tool(
        "video_edit_create_project",
        "Create or resume one rendered cloud-edit project from uploaded workflow assets.",
        {"workflow_id": {"type": "string"}, "user_prompt": {"type": "string", "maxLength": 512}},
        ["workflow_id"],
    ),
    _tool(
        "video_edit_wait_project",
        "Poll a rendered project for a bounded interval. Repeat with the same workflow when continue_required is true.",
        {"workflow_id": {"type": "string"}, "max_wait_seconds": {"type": "integer", "minimum": 15, "maximum": 480}},
        ["workflow_id"],
    ),
    _tool(
        "video_edit_download_result",
        "Download a completed rendered result into the active agent output directory and persist the resume checkpoint.",
        {"workflow_id": {"type": "string"}, "filename": {"type": "string", "maxLength": 128}},
        ["workflow_id"],
    ),
    _tool(
        "video_edit_proactive_resolve",
        "Resolve the server-owned weekly-memory manifest into a private workflow without exposing source paths to the model.",
        {"manifest_id": {"type": "string"}, "task_id": {"type": "string"}},
        ["manifest_id", "task_id"],
    ),
    _tool(
        "video_edit_proactive_report",
        "Report a successfully downloaded weekly-memory result to local-server exactly once.",
        {"workflow_id": {"type": "string"}},
        ["workflow_id"],
    ),
]
