---
name: authorized-connectors
description: "Use the user's already-authorized Zettlab apps such as GitHub, Linear, and Notion through Connector MCP tools."
version: 1.0.0
author: Zettlab
license: proprietary
metadata:
  hermes:
    tags: [connectors, zettlab, github, linear, notion, oauth, authorized-apps]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Authorized Zettlab Connectors

Use this skill when the user asks the agent to work with an app they have already authorized in Zettlab Connectors, especially GitHub, Linear, or Notion.

The user's OAuth authorization is managed by Zettlab. Do not ask for personal API keys. Do not read `GITHUB_TOKEN`, `GH_TOKEN`, `LINEAR_API_KEY`, `NOTION_API_KEY`, or provider tokens from environment variables. Do not call GitHub REST, Linear GraphQL, or Notion REST directly with `curl`. Use the connector MCP tools exposed to the agent.

## Tool Availability Rule

Before acting, use the tool surface available in the current session:

- If `linear.*` tools are available, use them for Linear requests.
- If `notion.*` tools are available, use them for Notion requests.
- If `github.*` tools are available, use them for GitHub account-backed repository requests.
- If the required tool is not available, explain that the connected app is not currently exposed to this Agent session. Ask the user to enable the app in Agent Connector permissions or the chat Connectors drawer, then retry in a fresh turn.
- Do not claim the app is not authorized unless the connector tool returns `not_connected`, `expired`, or `revoked`.

## GitHub Tools

Use these tools for account-backed GitHub repository data:

- `github.list_repos`
- `github.list_issues`
- `github.list_commits`
- `github.create_issue`

Use `github.list_repos` when the target repository is unclear. Use `github.list_issues` or `github.list_commits` only after the repo is known, and use `github.create_issue` only when the user clearly asks to create one. For local repository work, use normal git tools instead of connector tools.

## Linear Tools

Use these tools for Linear:

- `linear.list_issues`
- `linear.create_issue`
- `linear.update_issue`

### List Linear Issues

Use `linear.list_issues` for discovery.

Example arguments:

```json
{
  "team_key": "ZET",
  "state": "In Progress",
  "first": 20
}
```

Rules:

- `team_key` is optional. Use it when the user mentions a team key.
- `state` is optional. Use it when the user asks for a workflow state.
- `first` defaults to a small page. Use 20 unless the user asks for more.
- If multiple accounts are connected and the tool reports `ambiguous_account`, ask the user which account alias to use, then retry with `account_alias`.

### Create Linear Issue

Use `linear.create_issue` only when the user has provided enough intent to create an issue.

Required:

- `team_id`
- `title`

Optional:

- `description`

Example:

```json
{
  "team_id": "TEAM_UUID",
  "title": "Fix connector tools exposure",
  "description": "Agent session should expose authorized Linear tools after policy is enabled."
}
```

If the user only gives a team key, first list issues for that team and use the returned team id if present. If no team id is available, ask a short clarification.

### Update Linear Issue

Use `linear.update_issue` when the user asks to change an existing issue.

Required:

- `issue_id`

Optional:

- `title`
- `description`
- `state_id`

Only send fields the user actually asked to change.

## Notion Tools

Use these tools for Notion:

- `notion.search`
- `notion.get_page`
- `notion.create_page`

### Search Notion

Use `notion.search` when the user asks to find pages, docs, databases, notes, specs, or wiki content.

Example:

```json
{
  "query": "connector v1",
  "page_size": 10
}
```

Rules:

- Keep the query concise.
- If results are empty, say the connected Notion workspace returned no visible pages. Do not assume authorization failed.
- Notion OAuth may be authorized but only for pages the user selected during Notion sharing.

### Get Notion Page

Use `notion.get_page` when you already have a `page_id`.

Example:

```json
{
  "page_id": "PAGE_ID"
}
```

### Create Notion Page

Use `notion.create_page` when the user asks to create a page.

Required:

- `title`
- one of `parent_page_id` or `data_source_id`

Optional:

- `body`

Example:

```json
{
  "parent_page_id": "PAGE_ID",
  "title": "Connector acceptance notes",
  "body": "Linear and Notion are authorized and ready for agent validation."
}
```

If the user does not provide a parent page or data source, search first for the likely workspace page. If still unclear, ask where to create the page.

## Error Handling

Connector tools return structured connector errors. Treat them as state, not generic API failure.

- `not_connected`: ask the user to connect the provider in Zettlab Connectors.
- `denied_by_agent_policy`: ask the user to enable the provider in this Agent's Connector permissions.
- `denied_by_chat_override`: ask the user to enable the provider in the chat Connectors drawer.
- `expired` or `revoked`: ask the user to reconnect the provider.
- `ambiguous_account`: ask which connected account alias to use.
- `provider_error`: summarize the provider error briefly and suggest retry or reconnect only if the error indicates auth.

## Response Style

Summarize results in user language. For Linear, include issue identifier, title, state, assignee, and URL when available. For Notion, include page title, type, and URL or page id when available.

Never expose OAuth tokens, connector IDs, raw authorization headers, or full provider payloads unless the user explicitly asks for debugging details.
