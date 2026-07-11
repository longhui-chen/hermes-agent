"""markdown_vault plugin — read-only agent access to the user's Obsidian/markdown vault.

Provides three tools (vault_list / vault_read / vault_search) that retrieve
notes through the local-server loopback file API. There is deliberately no
write/move/delete surface — read-only is a property of the tool set, not of
instructions (spec 2026-07-08 local-data-access §4.6, D9/T7).

The profile that mounts this plugin should NOT also expose terminal /
write_file / execute_code / delegate_task etc.; withholding those at the
profile/toolset layer is what makes the whole surface read-only (skills and
skills_guard cannot enforce that at runtime). See SKILL.md.
"""

from __future__ import annotations

import logging

from plugins.markdown_vault.tools import (
    VAULT_LIST_SCHEMA,
    VAULT_READ_SCHEMA,
    VAULT_SEARCH_SCHEMA,
    check_vault_requirements,
    handle_vault_list,
    handle_vault_read,
    handle_vault_search,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    ("vault_list",   VAULT_LIST_SCHEMA,   handle_vault_list,   "📇"),
    ("vault_read",   VAULT_READ_SCHEMA,   handle_vault_read,   "📖"),
    ("vault_search", VAULT_SEARCH_SCHEMA, handle_vault_search, "🔎"),
)


def register(ctx) -> None:
    """Register the read-only vault tools. Called once by the plugin loader
    when the plugin is enabled via ``plugins.enabled`` in config.yaml."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="markdown_vault",
            schema=schema,
            handler=handler,
            check_fn=check_vault_requirements,
            emoji=emoji,
        )
    logger.info("markdown_vault: registered %d read-only tools", len(_TOOLS))
