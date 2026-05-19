---
name: zettlab-notion
description: "Notion via Zettlab Connectors: search, read page metadata, and create pages with the user's authorized Notion workspace."
version: 1.1.0
author: Zettlab
license: MIT
prerequisites:
  tools: [notion.search, notion.get_page, notion.create_page]
metadata:
  hermes:
    tags: [Notion, Connectors, Productivity, Notes, Knowledge Base]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Zettlab Notion

Use this skill when the user asks to search, inspect, or create Notion pages through Zettlab Connectors.

This skill does not own Notion authorization. Do not ask for a personal API key, do not read local Notion key environment variables, and do not call Notion REST directly. The user's account-level OAuth connection, Agent connector policy, and Chat session override decide whether the Notion tools are visible.

## Required Connector Tools

- `notion.search` - search pages and databases visible to the connected Notion workspace.
- `notion.get_page` - retrieve Notion page properties by page id.
- `notion.create_page` - create a Notion page under a page or data source.

If any required tool is unavailable, stop and explain the connector state from the tool/skill readiness result. Typical cases:

- `not_connected`: ask the user to connect Notion in Connectors.
- `denied_by_agent_policy`: ask the user to enable Notion for this Agent.
- `denied_by_chat_override`: ask the user to enable Notion in this chat's Connectors drawer.
- `expired` or `revoked`: ask the user to reconnect Notion.
- `ambiguous_account`: ask which Notion account or workspace to use.

## Usage Patterns

### Search Pages And Databases

Use `notion.search` first for discovery. Keep queries concise.

```json
{
  "query": "connector v1",
  "page_size": 10
}
```

Use `page_size` 10 by default unless the user asks for more. If results are empty, say the connected Notion workspace returned no visible pages. Do not assume authorization failed. Notion OAuth may be active while only the pages selected during sharing are visible.

### Get Page Properties

Use `notion.get_page` when a page id is already known from search results or user input.

```json
{
  "page_id": "PAGE_ID"
}
```

This retrieves page metadata/properties. Do not promise full block content unless the connector tool returns it.

### Create Page

Before creating, make sure the user has provided a title and a destination. The destination must be one of `parent_page_id` or `data_source_id`.

```json
{
  "parent_page_id": "PAGE_ID",
  "title": "Connector acceptance notes",
  "body": "Linear and Notion are authorized and ready for agent validation."
}
```

If the user does not provide a parent page or data source, search for the likely destination first. If still unclear, ask a short clarification.

## Response Style

Summarize the Notion result in user language: page title, type/object when present, URL or page id when available, and what was created or found. For write actions, confirm what changed and include the page URL or id if returned. Do not expose OAuth tokens, connector IDs, or raw provider payloads unless the user is debugging and specifically asks for them.
