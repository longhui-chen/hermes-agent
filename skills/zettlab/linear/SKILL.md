---
name: zettlab-linear
description: "Linear via Zettlab Connectors: list, create, and update issues with the user's authorized Linear account."
version: 1.1.0
author: Zettlab
license: MIT
prerequisites:
  tools: [linear.list_issues, linear.create_issue, linear.update_issue]
metadata:
  hermes:
    tags: [Linear, Connectors, Project Management, Issues, Productivity]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Zettlab Linear

Use this skill when the user asks to inspect, create, or update Linear issues through Zettlab Connectors.

This skill does not own Linear authorization. Do not ask for a personal API key, do not read Linear key environment variables, and do not call Linear GraphQL or REST directly. The user's account-level OAuth connection, Agent connector policy, and Chat session override decide whether the Linear tools are visible.

## Required Connector Tools

- `linear.list_issues` - list issues visible to the connected Linear account.
- `linear.create_issue` - create an issue in a Linear team.
- `linear.update_issue` - update issue title, description, or workflow state.

If any required tool is unavailable, stop and explain the connector state from the tool/skill readiness result. Typical cases:

- `not_connected`: ask the user to connect Linear in Connectors.
- `denied_by_agent_policy`: ask the user to enable Linear for this Agent.
- `denied_by_chat_override`: ask the user to enable Linear in this chat's Connectors drawer.
- `expired` or `revoked`: ask the user to reconnect Linear.
- `ambiguous_account`: ask which Linear account to use.

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

Use `team_key` when the user names a team key. Use `state` when the user asks for a specific workflow state. Use `account_alias` only when the user selected a specific connected account or the runtime reports an ambiguous account.

### Create Issue

Before creating, make sure the user has provided a team and title. If only a team key is known, list issues or ask a short clarification when the exact `team_id` is unavailable.

```json
{
  "team_id": "TEAM_UUID",
  "title": "Fix connector policy refresh",
  "description": "Chat drawer count should update after returning from connector authorization."
}
```

### Update Issue

Use an identifier or UUID in `issue_id`. Only send fields the user asked to change.

```json
{
  "issue_id": "ZET-275",
  "title": "Connectors V1 acceptance",
  "description": "Updated acceptance checklist",
  "state_id": "STATE_UUID"
}
```

For status changes, ask for or discover the target workflow state ID before calling `linear.update_issue`.

## Response Style

Summarize the Linear result in user language: issue identifier, title, state, assignee if present, and URL if returned. For write actions, confirm what changed and include the issue URL. Do not expose OAuth tokens, connector IDs, or raw provider payloads unless the user is debugging and specifically asks for them.
