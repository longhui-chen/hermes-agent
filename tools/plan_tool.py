#!/usr/bin/env python3
# zettlab-overlay(H1-B2b): present_plan 计划卡工具整文件为 fork 新增，B2b 删文件; upstream: none
"""
Plan Tool Module - Structured Plan Presentation

Allows the agent to present a structured execution plan to the user before
starting complex multi-step tasks. The agent calls this tool to display the
plan and then waits for the user's confirmation before proceeding.

Wire contract (hermes.plan SSE event):
    {
        "type": "hermes.plan",
        "plan_id": "a1b2c3d4e5f6",
        "title": "...",
        "groups": [
            {"icon": "emoji", "label": "group name", "count": 3, "items": ["step1", "step2", ...]}
        ]
    }

plan_id links the plan card to the todo items seeded from it (each seeded
item carries the same plan_id + its group_index) so clients can render both
as one evolving card. Optional field — old clients ignore it.

On the zet_agent platform, plan_emit_callback is injected by
ZetAgentAdapter._create_agent (same mechanism as clarify_callback / todo_emit_callback).
In non-zet-agent contexts (CLI, api_server) the callback is None and the tool
returns the plan as a formatted text block so the agent can still describe it.
"""

import json
import uuid
from typing import List, Optional, Dict, Any, Callable, Tuple

# 内存上限（HR-1）：LLM 可能吐超大计划，封顶防止单次 present_plan 撑爆内存 / 上下文。
_MAX_GROUPS = 20
_MAX_ITEMS_PER_GROUP = 50
_MAX_ITEM_LEN = 500
PLAN_PRESENTED_RESULT = (
    "Plan presented to user. "
    "A task list has been seeded from this plan (see the todo result below). "
    "Stop and wait for the user's confirmation before executing. After the "
    "user confirms, carry out the plan and update item statuses with the "
    "`todo` tool using merge=true. Do NOT rebuild the list from scratch."
)


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
    auto_execute: bool = False,
) -> str:
    """Present a structured execution plan to the user (legacy string return).

    Thin wrapper over :func:`present_plan_with_meta` for callers that only
    need the tool-result string (registry handler, tests).
    """
    result, _meta = present_plan_with_meta(
        title=title,
        groups=groups,
        callback=callback,
        auto_execute=auto_execute,
    )
    return result


def present_plan_with_meta(
    title: str,
    groups: List[Dict[str, Any]],
    callback: Optional[Callable] = None,
    auto_execute: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Present a structured execution plan to the user.

    Args:
        title:    Short title describing the overall goal.
        groups:   List of step groups. Each group:
                  {"icon": "emoji", "label": "group name", "count": int, "items": ["step1", ...]}
        callback: Platform-provided callable(title, groups, plan_id) -> None.
                  Injected by the agent runner on platforms that support
                  the hermes.plan SSE event (zet_agent). When None the
                  plan is returned as formatted text instead.

    Returns:
        (result_str, meta): meta 仅在 callback 路径成功 emit 卡片时非 None，
        形如 {"plan_id", "title", "groups"} —— 供 agent loop 用同一份计划骨架
        播种 TodoStore（见 agent/plan_seeding.py），保证「计划 ↔ 任务清单」的
        层级映射由代码保证而非提示词约定。
    """
    if not title or not title.strip():
        return json.dumps(
            {"error": "title is required for present_plan"},
            ensure_ascii=False,
        ), None

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
        if not items:
            continue
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

    if not cleaned_groups:
        return json.dumps(
            {"error": "at least one non-empty plan group is required"},
            ensure_ascii=False,
        ), None

    if callback is not None:
        # zet_agent：推 hermes.plan SSE → App 渲染结构化计划卡。
        # plan_id 是「计划 ↔ 任务清单」的锚：App 按它把两者渲染成同一张卡，
        # 播种的 todo 条目也携带它（跨 turn / 冷启动 / 历史回放归并的唯一键）。
        plan_id = uuid.uuid4().hex[:12]

        # 卡片延迟到持久化成功后再推（codex P1）：在这里直接 emit 会让 App 拿到
        # 一张只存在于内存的可确认计划卡——随后 state.db 写失败时 turn 走
        # session_persistence_failed，播种也没落盘，用户确认后的下一轮历史里
        # 没有可配对的 plan/todo 状态，合一卡与执行上下文一起丢。emit 交给
        # agent/plan_seeding 在 flush 成功后调用。
        def _emit_plan_card() -> None:
            try:
                callback(title, cleaned_groups, plan_id)
            except TypeError:
                # 旧签名 callback(title, groups)：测试替身 / 未升级平台兼容。
                callback(title, cleaned_groups)

        meta = {
            "plan_id": plan_id,
            "title": title,
            "groups": cleaned_groups,
            "emit": _emit_plan_card,
        }
        if auto_execute:
            # 旧 App 兼容路径（metadata opt-in 直跑）：计划卡只读展示，agent 在
            # 同一 turn 直接继续执行。任务清单已由播种机制按计划骨架创建，
            # agent 只更新状态，不再自行另建清单（详略错位的根源，FND-004）。
            return (
                "Plan presented to the user's App as a read-only card. "
                "Auto-execute is enabled: start carrying out the plan now in "
                "this same turn. Do NOT ask the user to confirm or say you are "
                "waiting — just proceed. A task list has already been created "
                "from this plan (see the todo result below); update item "
                "statuses with the `todo` tool using merge=true as you work. "
                "Do NOT rebuild the list from scratch."
            ), meta
        return PLAN_PRESENTED_RESULT, meta

    # 无 callback（CLI / messaging / api_server，无确认卡）：返回格式化计划文本，
    # 让 agent 能把计划完整呈现给用户，再停下等确认——否则计划内容丢失且 agent 空等。
    # 此分支不播种（delegate 子 agent / CLI 维持现状，方案 §4 边界表）。
    plan_text = _format_plan_text(title, cleaned_groups)
    if auto_execute:
        # delegated child（delegate_tool 强制 _zet_agent_plan_auto_execute）等
        # 无 UI 场景：没有用户可回复确认，返回「等确认」文案会让子 agent 空等
        # 耗尽迭代预算——保留计划文本作上下文，指示立即继续执行。
        return (
            plan_text
            + "\n\nAuto-execute is enabled and there is no user available to "
            "confirm: start carrying out the plan now in this same turn. Do "
            "NOT wait for a reply — just proceed."
        ), None
    # ⛔ 这里只能放**给模型看的指令**，不许放面向用户的话术（ZET-3140）。
    # 旧文案是第一人称、直接对用户说话的英文句子（"…before I proceed."，还带
    # "go" / "confirm" 这种示例回复词），模型会把它原样复述出去 —— 中文会话里
    # 就蹦英文。另外三条分支附的都是祈使式内部指令，模型不会念给用户。
    # ⚠️ 砍的作用域**刚好等于**「英文用户话术」：原句同时承担「需要用户确认」
    # 这个语义，那部分必须原样保留，否则计划会不等确认就被执行。
    return (
        plan_text
        + "\n\nThis plan is awaiting the user's confirmation. Ask them to "
        "confirm it, phrased in the language of the conversation, and do NOT "
        "start executing any item until they do."
    ), None


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
                "minItems": 1,
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
                            "minItems": 1,
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
        auto_execute=kw.get("auto_execute", False),
    ),
    check_fn=check_plan_requirements,
    emoji="📋",
)
