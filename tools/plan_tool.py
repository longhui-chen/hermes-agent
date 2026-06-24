#!/usr/bin/env python3
"""
Plan Tool Module - Structured Plan Presentation

Allows the agent to present a structured execution plan to the user before
starting complex multi-step tasks. The agent calls this tool to display the
plan and then waits for the user's confirmation before proceeding.

Wire contract (hermes.plan SSE event):
    {
        "type": "hermes.plan",
        "title": "...",
        "groups": [
            {"icon": "emoji", "label": "group name", "count": 3, "items": ["step1", "step2", ...]}
        ]
    }

On the zet_agent platform, plan_emit_callback is injected by
ZetAgentAdapter._create_agent (same mechanism as clarify_callback / todo_emit_callback).
In non-zet-agent contexts (CLI, api_server) the callback is None and the tool
returns the plan as a formatted text block so the agent can still describe it.
"""

import json
from typing import List, Optional, Dict, Any, Callable


def present_plan(
    title: str,
    groups: List[Dict[str, Any]],
    callback: Optional[Callable] = None,
) -> str:
    """
    Present a structured execution plan to the user.

    Args:
        title:    Short title describing the overall goal.
        groups:   List of step groups. Each group:
                  {"icon": "emoji", "label": "group name", "count": int, "items": ["step1", ...]}
        callback: Platform-provided callable(title, groups) -> None.
                  Injected by the agent runner on platforms that support
                  the hermes.plan SSE event (zet_agent). When None the
                  plan is returned as formatted text instead.

    Returns:
        A short instruction telling the agent to stop and wait for user
        confirmation before executing the plan.
    """
    if not title or not title.strip():
        return json.dumps(
            {"error": "title is required for present_plan"},
            ensure_ascii=False,
        )

    title = title.strip()

    # Normalise groups: ensure required fields, clip oversized items lists.
    cleaned_groups: List[Dict[str, Any]] = []
    for g in (groups or []):
        if not isinstance(g, dict):
            continue
        items = [str(i).strip() for i in (g.get("items") or []) if str(i).strip()]
        label = str(g.get("label", "")).strip() or "Steps"
        icon = str(g.get("icon", "")).strip() or ""
        count = int(g.get("count") or len(items)) if g.get("count") is not None else len(items)
        cleaned_groups.append({
            "icon": icon,
            "label": label,
            "count": count,
            "items": items,
        })

    if callback is not None:
        try:
            callback(title, cleaned_groups)
        except Exception:
            pass

    return (
        "Plan presented to user. "
        "Stop and wait for the user's confirmation before executing."
    )


def check_plan_requirements() -> bool:
    """Present-plan tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

PLAN_SCHEMA = {
    "name": "present_plan",
    "description": (
        "Present a structured execution plan to the user before starting a "
        "complex multi-step task. Call this tool when the task involves 3 or "
        "more distinct phases, significant data mutation, or irreversible "
        "actions. After calling this tool, STOP and wait for the user's "
        "confirmation message before doing any actual work.\n\n"
        "Use groups to organise steps into logical phases (e.g. 'Analysis', "
        "'Implementation', 'Verification'). Each group should have 1-5 items "
        "describing concrete steps.\n\n"
        "Do NOT use this tool for simple single-step requests or routine "
        "lookups — only for plans that genuinely benefit from upfront review."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title (≤80 chars) summarising the overall goal.",
            },
            "groups": {
                "type": "array",
                "description": "Ordered list of step groups forming the plan.",
                "items": {
                    "type": "object",
                    "properties": {
                        "icon": {
                            "type": "string",
                            "description": "A single emoji representing the group theme.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short group name (e.g. 'Analysis', 'Setup').",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of items in this group.",
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete steps in this group (1-5 items).",
                        },
                    },
                    "required": ["icon", "label", "count", "items"],
                },
            },
        },
        "required": ["title", "groups"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="present_plan",
    toolset="plan",
    schema=PLAN_SCHEMA,
    handler=lambda args, **kw: present_plan(
        title=args.get("title", ""),
        groups=args.get("groups", []),
        callback=kw.get("callback"),
    ),
    check_fn=check_plan_requirements,
    emoji="📋",
)
