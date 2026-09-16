# zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
"""Live session context-window breakdown for UI surfaces.

Reports provider-measured capacity separately from UTF-8 payload composition.
Payload bytes are diagnostic only, never a substitute for model token usage.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _split_tools(tools: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    builtin: List[dict] = []
    mcp: List[dict] = []
    subagent: List[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> Tuple[str, str]:
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
    from agent.model_metadata import _wire_message_shadow
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)

    # zettlab-overlay(context-budget): category attribution is literal UTF-8 size, never token occupancy; upstream: none
    def text_bytes(text):
        return len(text.encode("utf-8"))

    def json_bytes(value):
        return sum(text_bytes(chunk) for chunk in json.JSONEncoder(ensure_ascii=False, default=str).iterencode(value))

    conversation_tokens = sum(json_bytes(_wire_message_shadow(m)) for m in messages or [])

    categories = [
        # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
        ("system_prompt", "System prompt", text_bytes(system_prompt_text)),
        ("tool_definitions", "Tool definitions", json_bytes(builtin_tools)),
        ("rules", "Rules", text_bytes(context)),
        ("skills", "Skills", text_bytes(skills_index)),
        ("mcp", "MCP", json_bytes(mcp_tools)),
        ("subagent_definitions", "Subagent definitions", json_bytes(subagent_tools)),
        ("memory", "Memory", text_bytes(memory_text)),
        ("conversation", "Conversation", conversation_tokens),
    ]

    estimated_total = sum(tokens for _, _, tokens in categories)

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    # zettlab-overlay(context-budget): missing usage is unknown, not the category byte total; upstream: none
    context_used = max(0, measured_used)
    if comp and getattr(comp, "awaiting_real_usage_after_compression", False) is True:
        context_used = 0
    context_percent = (
        max(0, min(100, round(context_used / context_max * 100)))
        if context_max
        else 0
    )

    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(category_id, "var(--ui-text-tertiary)"),
                "id": category_id,
                "label": label,
                # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
                "tokens": 0,  # Legacy field: category tokens are not measured.
                "bytes": tokens,
            }
            for category_id, label, tokens in categories
            if tokens > 0
        ],
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
        "estimated_total": 0,
        "total_bytes": estimated_total,
        "size_unit": "bytes",
        "context_measurement": "provider" if context_used > 0 else "unknown",
        "model": getattr(agent, "model", "") or "",
    }


# ── /context rendering (CLI + gateway) ──────────────────────────────────────
#
# Pure text renderers over the payload above. The CLI shows a glyph block-grid
# plus a category table; the gateway uses the same table without the grid
# (proportional monospace is not guaranteed on messaging platforms).

_CATEGORY_GLYPHS = {
    "system_prompt": "■",
    "tool_definitions": "▣",
    "rules": "▩",
    "skills": "▤",
    "mcp": "▥",
    "subagent_definitions": "▦",
    "memory": "▧",
    "conversation": "▨",
}
_FREE_GLYPH = "·"
_GRID_COLUMNS = 20
_GRID_ROWS = 5  # 100 cells → 1 cell per percent of the context window

# Human-readable tables cap the expanded listings; nothing is dropped from
# the underlying data.
_DETAILS_TABLE_LIMIT = 15


def compute_context_details(agent: Any) -> Dict[str, Any]:
    """Expanded per-skill / per-toolset cost listing for ``/context all``.

    Reuses the ``hermes prompt-size`` attribution mechanism (PR #66656):
    per-skill index-line bytes parsed from the live ``<available_skills>``
    block, and per-toolset schema bytes attributed via the tool registry's
    canonical tool→toolset map. Byte figures are converted to the same
    chars/4 token heuristic the categories above use.
    """
    from hermes_cli.prompt_size import (
        _compute_skills_breakdown,
        _compute_toolsets_breakdown,
    )
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_block = skills_match.group(0) if skills_match else ""

    skills: List[Dict[str, Any]] = []
    if skills_block:
        for entry in _compute_skills_breakdown(skills_block):
            skills.append({
                "name": entry.get("name", ""),
                # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
                "index_tokens": 0,
                "index_bytes": entry.get("index_line_bytes") or 0,
                "skill_md_tokens": None,
                "skill_md_bytes": entry.get("skill_md_bytes"),
            })

    toolsets: List[Dict[str, Any]] = []
    tools = list(getattr(agent, "tools", None) or [])
    if tools:
        for group in _compute_toolsets_breakdown(tools):
            toolsets.append({
                "toolset": group.get("toolset", ""),
                "tool_count": int(group.get("tool_count", 0) or 0),
                # zettlab-overlay(context-budget): keep literal payload size separate from provider token capacity; upstream: none
                "schema_tokens": 0,
                "schema_bytes": group.get("json_bytes") or 0,
            })

    # zettlab-overlay(context-budget): detail attribution retains byte units; upstream: none
    return {"skills": skills, "toolsets": toolsets, "size_unit": "bytes"}


def render_context_grid(payload: Dict[str, Any]) -> List[str]:
    """Render the payload as a Claude Code-style glyph block grid.

    100 cells (5×20), each one percent of the model context window. Categories
    fill in declaration order; the remainder renders as free space.
    """
    # zettlab-overlay(context-budget): occupancy grid must not map payload bytes onto model tokens; upstream: none
    if payload.get("context_measurement") == "unknown":
        return ["Context usage: not measured"]
    if payload.get("size_unit") == "bytes":
        used = min(100, max(0, int(payload.get("context_percent") or 0)))
        cells = ["▨"] * used + [_FREE_GLYPH] * (100 - used)
        return [" ".join(cells[i:i + _GRID_COLUMNS]) for i in range(0, 100, _GRID_COLUMNS)]
    context_max = int(payload.get("context_max") or 0)
    categories = payload.get("categories") or []
    total_cells = _GRID_COLUMNS * _GRID_ROWS

    cells: List[str] = []
    if context_max > 0:
        for cat in categories:
            tokens = int(cat.get("tokens") or 0)
            n = round(tokens / context_max * total_cells)
            if tokens > 0 and n == 0:
                n = 1  # never render a nonzero category as invisible
            glyph = _CATEGORY_GLYPHS.get(str(cat.get("id") or ""), "▪")
            cells.extend([glyph] * n)
        cells = cells[:total_cells]
    cells.extend([_FREE_GLYPH] * (total_cells - len(cells)))

    return [
        " ".join(cells[row * _GRID_COLUMNS:(row + 1) * _GRID_COLUMNS])
        for row in range(_GRID_ROWS)
    ]


def render_context_category_lines(payload: Dict[str, Any]) -> List[str]:
    """Render the 'Estimated usage by category' table as plain-text lines."""
    # zettlab-overlay(context-budget): byte composition is not a token-capacity percentage; upstream: none
    if payload.get("size_unit") == "bytes":
        return ["Payload size by category (bytes, not token occupancy)"] + [
            f"  {cat['label']}: {cat.get('bytes', 0):,} bytes"
            for cat in payload.get("categories", [])
        ]
    categories = payload.get("categories") or []
    context_max = int(payload.get("context_max") or 0)
    estimated_total = int(payload.get("estimated_total") or 0)
    denom = context_max or estimated_total

    lines = ["Estimated usage by category"]
    if not categories:
        lines.append("  (no data yet — send a message first)")
        return lines

    width = max(len(str(cat.get("label") or "")) for cat in categories)
    width = max(width, len("Free space"))
    for cat in categories:
        tokens = int(cat.get("tokens") or 0)
        glyph = _CATEGORY_GLYPHS.get(str(cat.get("id") or ""), "▪")
        pct = tokens / denom * 100 if denom else 0.0
        label = str(cat.get("label") or cat.get("id") or "")
        lines.append(f"{glyph} {label:<{width}} {tokens:>9,} tokens {pct:>5.1f}%")
    if context_max > 0:
        free = max(0, context_max - estimated_total)
        pct = free / context_max * 100
        lines.append(f"{_FREE_GLYPH} {'Free space':<{width}} {free:>9,} tokens {pct:>5.1f}%")
    return lines


def render_context_details_lines(details: Dict[str, Any]) -> List[str]:
    """Render the expanded ``/context all`` per-skill / per-toolset tables."""
    # zettlab-overlay(context-budget): report source file and schema bytes without conversion; upstream: none
    if details.get("size_unit") == "bytes":
        lines = ["Source sizes (bytes)"]
        for group in details.get("toolsets", [])[:_DETAILS_TABLE_LIMIT]:
            lines.append(f"  {group['toolset']}: {group.get('schema_bytes', 0):,} bytes")
        for item in details.get("skills", [])[:_DETAILS_TABLE_LIMIT]:
            lines.append(f"  {item['name']}: index {item.get('index_bytes', 0)} bytes, SKILL.md {item.get('skill_md_bytes', 'unknown')} bytes")
        return lines
    lines: List[str] = []

    toolsets = details.get("toolsets") or []
    if toolsets:
        lines.append("Toolsets by schema cost (largest first)")
        for group in toolsets[:_DETAILS_TABLE_LIMIT]:
            lines.append(
                f"  {group['toolset']:<24} {group['tool_count']:>3} tools"
                f" {group['schema_tokens']:>8,} tokens"
            )
        remaining = len(toolsets) - _DETAILS_TABLE_LIMIT
        if remaining > 0:
            lines.append(f"  … and {remaining} more")

    skills = details.get("skills") or []
    if skills:
        if lines:
            lines.append("")
        lines.append("Skills by cost (index = always-on; SKILL.md = cost when loaded)")
        for entry in skills[:_DETAILS_TABLE_LIMIT]:
            name = str(entry.get("name") or "")
            if len(name) > 28:
                name = name[:27] + "…"
            md = entry.get("skill_md_tokens")
            md_str = f"{md:>8,}" if md is not None else f"{'n/a':>8}"
            lines.append(
                f"  {name:<28} index {entry['index_tokens']:>6,}"
                f"  SKILL.md {md_str} tokens"
            )
        remaining = len(skills) - _DETAILS_TABLE_LIMIT
        if remaining > 0:
            lines.append(f"  … and {remaining} more")

    return lines


def render_context_breakdown_lines(
    payload: Dict[str, Any],
    *,
    details: Optional[Dict[str, Any]] = None,
    grid: bool = True,
) -> List[str]:
    """Render the full /context view as plain-text lines.

    ``grid=True`` (CLI) prepends the glyph block grid; the gateway passes
    ``grid=False`` and keeps its own gauge. ``details`` (from
    :func:`compute_context_details`) appends the expanded listings.
    """
    lines: List[str] = []
    if grid:
        lines.extend(render_context_grid(payload))
        lines.append("")
    lines.extend(render_context_category_lines(payload))

    # zettlab-overlay(context-budget): expose unmeasured capacity explicitly; upstream: none
    if payload.get("context_measurement") == "unknown":
        lines.append("Context window usage: not measured; waiting for provider usage.")
        if details is not None:
            lines.extend(render_context_details_lines(details))
        return lines
    context_max = int(payload.get("context_max") or 0)
    context_used = int(payload.get("context_used") or 0)
    if context_max > 0:
        pct = int(payload.get("context_percent") or 0)
        lines.append("")
        lines.append(
            f"Context window: {context_used:,} / {context_max:,} tokens ({pct}%)"
        )

    if details is not None:
        detail_lines = render_context_details_lines(details)
        if detail_lines:
            lines.append("")
            lines.extend(detail_lines)
    else:
        lines.append("")
        lines.append("Use /context all for per-skill and per-toolset costs.")
    return lines
