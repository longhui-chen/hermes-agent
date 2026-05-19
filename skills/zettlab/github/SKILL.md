---
name: zettlab-github
description: "GitHub via Zettlab Connectors: list repositories, inspect issues and commits, and create issues with the user's authorized GitHub account."
version: 1.0.0
author: Zettlab
license: MIT
prerequisites:
  tools: [github.list_repos, github.list_issues, github.list_commits, github.create_issue]
metadata:
  hermes:
    tags: [GitHub, Connectors, Repositories, Issues, Commits]
    related_skills: [github-issues, github-repo-management, github-pr-workflow, github-code-review]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Zettlab GitHub

Use this skill when the user asks to inspect or create GitHub repository data through Zettlab Connectors.

This skill does not own GitHub authorization. Do not ask for a personal access token, do not read local GitHub token environment variables, and do not call the GitHub REST API directly. The user's account-level OAuth connection, Agent connector policy, and Chat session override decide whether the GitHub tools are visible.

For local git operations, such as reading the current worktree, creating commits, or inspecting local diffs, use normal git tools and the relevant local GitHub workflow skills. For account-backed GitHub API data, use the connector tools below.

## Required Connector Tools

- `github.list_repos` - list repositories visible to the connected GitHub account.
- `github.list_issues` - list issues for a repository.
- `github.list_commits` - list recent commits for a repository.
- `github.create_issue` - create an issue in a repository.

If any required tool is unavailable, stop and explain the connector state from the tool/skill readiness result. Typical cases:

- `not_connected`: ask the user to connect GitHub in Connectors.
- `denied_by_agent_policy`: ask the user to enable GitHub for this Agent.
- `denied_by_chat_override`: ask the user to enable GitHub in this chat's Connectors drawer.
- `expired` or `revoked`: ask the user to reconnect GitHub.
- `ambiguous_account`: ask which GitHub account to use.

## Usage Patterns

### List Repositories

Use `github.list_repos` when the user asks what repositories are available or when you need to discover the owner/repo name.

```json
{
  "per_page": 20
}
```

Use `account_alias` only when the user selected a specific connected account or the runtime reports an ambiguous account.

### List Issues

Use `github.list_issues` when the user asks about issues in a repository.

```json
{
  "repo": "owner/repo",
  "state": "open",
  "per_page": 20
}
```

Rules:

- `repo` can be `owner/name`. If only the repo name is known, use `owner` separately or discover it first.
- `state` can be `open`, `closed`, or `all`; default to `open`.
- Keep `per_page` modest unless the user asks for more.

### List Commits

Use `github.list_commits` to inspect recent commits.

```json
{
  "repo": "owner/repo",
  "sha": "main",
  "per_page": 10
}
```

`sha` is optional and can be a branch, tag, or commit SHA.

### Create Issue

Use `github.create_issue` only when the user clearly asks to create an issue.

Required:

- `repo`
- `title`

Optional:

- `owner`
- `body`
- `labels`

Example:

```json
{
  "repo": "owner/repo",
  "title": "Fix connector skill routing",
  "body": "Agent should use Zettlab connector-backed GitHub tools instead of asking for local GitHub tokens.",
  "labels": ["bug"]
}
```

If the target repository is ambiguous, ask a short clarification before creating.

## Response Style

Summarize the GitHub result in user language. For repositories, include repo full name and URL when returned. For issues, include number/title/state/URL when available. For commits, include short SHA, message, author, and date when returned. For write actions, confirm what was created and include the issue URL. Do not expose OAuth tokens, connector IDs, raw authorization headers, or full provider payloads unless the user is debugging and specifically asks for them.
