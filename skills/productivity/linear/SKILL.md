---
name: linear
description: "Linear direct API helper: inspect, create, and update Linear issues with a LINEAR_API_KEY in non-Zettlab environments."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  environment: LINEAR_API_KEY
metadata:
  hermes:
    tags: [Linear, Project Management, Issues, Productivity]
    related_skills: []
---

# Linear - Direct API Mode

Use this skill when the user explicitly wants to work with Linear through a personal Linear API key or when the environment is not using Zettlab Connectors.

In Zettlab profiles, prefer the official connector preset skill `zettlab-linear` from `zettlab-presets`. That preset uses account-level OAuth, Agent connector policy, and Chat connector overrides. This bundled Hermes skill is only the direct API fallback.

## Authentication

The helper reads `LINEAR_API_KEY` from the environment and calls Linear GraphQL with the personal API key header format:

```bash
export LINEAR_API_KEY=lin_api_...
```

Never print the key or copy it into user-visible output. If the key is missing, explain that direct Linear API mode needs `LINEAR_API_KEY`.

## Helper

Use the bundled zero-dependency helper:

```bash
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py <command> [args...]
```

Common read commands:

```bash
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py whoami
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py list-teams
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py list-states --team ENG
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py list-issues --team ENG --status "In Progress" --limit 20
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py get-issue ENG-42
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py search-issues "connector policy"
```

Common write commands:

```bash
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py create-issue --team ENG --title "Fix connector policy refresh" --description "Chat drawer count should update after authorization."
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py update-issue ENG-42 --title "Updated title"
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py update-status ENG-42 Done
python3 ~/.hermes/skills/productivity/linear/scripts/linear_api.py add-comment ENG-42 "Validated in staging."
```

Only run write commands when the user clearly asked to create or change something. If the target team, issue, title, or workflow state is ambiguous, ask a short clarification first.

## Response Style

Summarize Linear results in user language: issue identifier, title, state, assignee if present, and URL when returned. For write actions, confirm the change and include the issue URL if the helper returns one.
