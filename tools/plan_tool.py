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

# 内存上限（HR-1）：LLM 可能吐超大计划，封顶防止单次 present_plan 撑爆内存 / 上下文。
_MAX_GROUPS = 20
_MAX_ITEMS_PER_GROUP = 50
_MAX_ITEM_LEN = 500


def _format_plan_text(title: str, groups: List[Dict[str, Any]]) -> str:
    """把结构化计划渲染成纯文本块（无 UI 卡片的平台用，如 CLI / messaging）。"""
    lines = [f"📋 {title}"]
    for g in groups:
        header = f"{g.get('icon', '')} {g.get('label', '')}".strip()
        lines.append(f"\n{header} ({g.get('count', 0)})")
        for item in g.get("items", []):
            lines.append(f"  • {item}")
    return "\n".join(lines)


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

    # Normalise + clip：封顶 groups / 每组 items 数 / 单条长度（HR-1，防超大计划）。
    cleaned_groups: List[Dict[str, Any]] = []
    for g in (groups or [])[:_MAX_GROUPS]:
        if not isinstance(g, dict):
            continue
        items = [
            str(i).strip()[:_MAX_ITEM_LEN]
            for i in (g.get("items") or [])
            if str(i).strip()
        ][:_MAX_ITEMS_PER_GROUP]
        label = str(g.get("label", "")).strip() or "Steps"
        icon = str(g.get("icon", "")).strip() or ""
        # count 一律以裁剪后的 items 实际条数为准（schema 里 count 可选；传了也忽略，
        # 避免「模型给的 count」与实际条目数不一致）。
        cleaned_groups.append({
            "icon": icon,
            "label": label,
            "count": len(items),
            "items": items,
        })

    if callback is not None:
        # zet_agent：推 hermes.plan SSE → App 渲染结构化确认卡。
        try:
            callback(title, cleaned_groups)
        except Exception:
            pass
        return (
            "Plan presented to user. "
            "Stop and wait for the user's confirmation before executing."
        )

    # 无 callback（CLI / messaging / api_server，无确认卡）：返回格式化计划文本，
    # 让 agent 能把计划完整呈现给用户，再停下等确认——否则计划内容丢失且 agent 空等。
    return (
        _format_plan_text(title, cleaned_groups)
        + "\n\nReview the plan above and reply to confirm (e.g. \"go\" / \"confirm\") before I proceed."
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
        "Before calling this tool, verify that all essential user input needed "
        "to execute the plan is already known. If essential information is "
        "missing, call `clarify` first instead. Never include collecting required "
        "user information or resolving a prerequisite decision as a plan step; "
        "the plan must be executable immediately after confirmation. Do not "
        "replace materially plan-changing missing inputs with generic assumptions. "
        "Personalized health, diet, and fitness plans require the user's baseline, "
        "goal, timeframe, and relevant constraints before presentation.\n\n"
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
                            "description": "Optional. Ignored if provided — count is derived from items.",
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete steps in this group (1-5 items).",
                        },
                    },
                    "required": ["icon", "label", "items"],
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
