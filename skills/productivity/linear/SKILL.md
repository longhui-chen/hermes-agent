---
name: linear
description: "Linear via Zettlab Connectors first: list, create, and update issues with the user's authorized Linear account."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  tools: [linear.list_issues, linear.create_issue, linear.update_issue]
metadata:
  hermes:
    tags: [Linear, Connectors, Project Management, Issues, Productivity]
    related_skills: [zettlab-linear, authorized-connectors]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Linear - Issue & Project Management

Use this skill when the user asks to inspect, create, or update Linear issues.

In Zettlab, Linear authorization belongs to Connectors. Use the user's already-authorized Linear account through the connector tools. Do not ask for a personal Linear API key, do not read `LINEAR_API_KEY`, and do not call Linear GraphQL directly unless the user explicitly asks for direct Linear API mode outside the Zettlab connector flow.

This skill should behave like `zettlab-linear`: account-level OAuth connection, Agent connector policy, and Chat session override decide whether the Linear tools are visible and callable.

## Required Connector Tools

- `linear.list_issues` - list issues visible to the connected Linear account.
- `linear.create_issue` - create an issue in a Linear team.
- `linear.update_issue` - update issue title, description, or workflow state.

If a required tool is unavailable, stop and explain the connector state from the tool or skill readiness result. Typical cases:

- `not_connected`: ask the user to connect Linear in Connectors.
- `denied_by_agent_policy`: ask the user to enable Linear for this Agent.
- `denied_by_chat_override`: ask the user to enable Linear in this chat's Connectors drawer.
- `expired` or `revoked`: ask the user to reconnect Linear.
- `ambiguous_account`: ask which Linear account to use.

Do not report "Linear is not authorized" just because a tool is hidden by Agent policy or chat override. Name the actual blocking layer when the runtime returns it.

## Usage Patterns

### List Issues

Use `linear.list_issues` first for discovery. Keep filters minimal unless the user gives specifics.

```json
{
  "team_key": "ENG",
  "state": "In Progress",
  "first": 20
}
```

Rules:

- `team_key` is optional. Use it when the user mentions a team key.
- `state` is optional. Use it when the user asks for a workflow state.
- `first` defaults to a small page. Use 20 unless the user asks for more.
- Use `account_alias` only when the user selected a specific connected account or the runtime reports `ambiguous_account`.

### Create Issue

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
  "title": "Fix connector policy refresh",
  "description": "Chat drawer count should update after returning from connector authorization."
}
```

If the user only gives a team key, first list issues for that team and use the returned team id if present. If no team id is available, ask a short clarification.

### Update Issue

Use `linear.update_issue` when the user asks to change an existing issue.

Required:

- `issue_id`

Optional:

- `title`
- `description`
- `state_id`

Example:

```json
{
  "issue_id": "ZET-275",
  "title": "Connectors V1 acceptance",
  "description": "Updated acceptance checklist",
  "state_id": "STATE_UUID"
}
```

For status changes, ask for or discover the target workflow state ID before calling `linear.update_issue`. Only send fields the user actually asked to change.

## Direct Linear API Fallback

The bundled `scripts/linear_api.py` helper is a non-Zettlab fallback for environments that are not using Connectors. Use it only when the user explicitly asks for direct Linear API mode or provides a personal API key workflow.

When using direct mode, the helper reads `LINEAR_API_KEY` and calls `https://api.linear.app/graphql`. This mode does not use the user's Zettlab connector authorization, Agent policy, or chat connector override.

## Response Style

Summarize the Linear result in user language: issue identifier, title, state, assignee if present, and URL if returned. For write actions, confirm what changed and include the issue URL. Do not expose OAuth tokens, connector IDs, raw authorization headers, or full provider payloads unless the user is debugging and specifically asks for them.
