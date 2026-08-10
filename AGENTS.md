# Hermes Agent — Agent Instructions

This file is the automatically loaded entry point for work in `hermes-agent`.
Keep it focused on rules that apply broadly. Detailed architecture, feature
inventories, command references, and tutorials belong in the linked docs or next
to the subsystem they describe.

**Never give up on the right solution.**

## Read order and authority

1. When this repository is checked out inside `zettlab-product-dev`, first read
   the parent `AGENTS.md` and its routed rules. Parent workspace rules still
   apply unless this file is more specific about Hermes.
2. Read this file completely.
3. Read only the task-specific documents from the routing table below.
4. For `apps/desktop/**`, also read [`apps/desktop/AGENTS.md`](./apps/desktop/AGENTS.md)
   and [`apps/desktop/DESIGN.md`](./apps/desktop/DESIGN.md).

Source code and tests are authoritative for current behavior. Documentation is
for invariants, intent, and navigation; do not preserve a stale implementation
merely because an old guide describes it.

## Review guidelines

- GitHub Codex review comments must use Simplified Chinese.
- Publish only `P0` / `P1` findings as GitHub review comments. Keep the severity
  labels unchanged; write the title, impact, and fix direction in Chinese.
- Keep code symbols, identifiers, paths, routes, schemas, error codes,
  environment variables, API names, and protocol names in their original form.
- Each finding must state a user-visible impact or system risk and an actionable
  fix direction.

## Zettlab Engineering Hard Rules

These rules apply even when this repository is cloned outside the monorepo. In
the monorepo, the [complete parent rule](https://github.com/zettlab/zettlab-product-dev/blob/main/.agents/rules/engineering-hard-rules.md)
is authoritative; update it first, then intentionally sync this mandatory
Hermes summary. Every code change, design, plan, spec, and PR must explicitly
assess all rules, including “no impact” where appropriate.

1. **Device memory budget:** Hermes shares roughly 2 GB RAM with other device
   services. No unbounded caches or histories, leaked processes/connections, full
   reads of large files, or duplicate model loads across profiles. Bound resident
   state with limits, LRU, TTL, paging, chunks, or streaming. Every profile child
   process needs an explicit RSS ceiling and reclamation policy; report expected
   steady-state RSS impact.
2. **Reliability and HA:** External dependencies need timeouts, bounded retries,
   and a degradation path. Long-running processes must be restartable. Prefer
   staging, atomic rename, backup, and recoverable state transitions. Explain how
   a feature fails, recovers, and degrades safely.
3. **Agent safety:** Tool, MCP, connector, skill, shell, filesystem, SQL, token,
   and user-data access must validate owner, scope, paths, and command allowlists.
   Never pass cloud user tokens to a local shell or interpolate untrusted LLM
   output directly into SQL, shell commands, or filesystem paths. Connectors must
   not default to full user-data scope. A PR adding a connector, tool, skill, or
   MCP server must state the worst harm it could cause.
4. **API compatibility:** Keep `/v1/chat/completions`, SSE events, message storage,
   and Zettlab extensions backward compatible. Add optional fields instead of
   removing or changing existing ones. Use capability negotiation or feature
   flags when consumers cannot update atomically, and cover contract changes with
   end-to-end verification across every consumer.
5. **Trade-off order:** reliability > security > performance. Any exception must
   be explicit, reviewed, degradable, and reversible.
6. **Device configuration source:** packaged device configuration comes only from
   `zpk/config/<repo>.yaml`. Do not introduce `config.example.yaml`,
   `config.board.yaml`, or repository-root `config.yaml` as device runtime
   sources. Build-bundle, remote-install, and OTA paths must use that same source;
   development differences belong in environment variables. If this checkout has
   no runtime YAML, use the owning packaging repository instead of inventing one.
7. **HR-T1 dependency freeze:** while the parent freeze is active, do not add,
   upgrade, or downgrade Microsoft/Azure dependency families (`Azure`,
   `Microsoft`, `@azure`, `@microsoft`, `azure-*`, `msal*`). Existing pinned
   versions may remain; removal is allowed.

## Task routing

| Task | Read before editing |
|---|---|
| Contribution setup, code style, tools, skills, dependencies, PR process | [`CONTRIBUTING.md`](./CONTRIBUTING.md) |
| Overall architecture and dependency direction | [`website/docs/developer-guide/architecture.md`](./website/docs/developer-guide/architecture.md) |
| Agent loop, budgets, tool execution, persistence | [`website/docs/developer-guide/agent-loop.md`](./website/docs/developer-guide/agent-loop.md) |
| Prompt/context-file assembly | [`website/docs/developer-guide/prompt-assembly.md`](./website/docs/developer-guide/prompt-assembly.md) |
| Compression or prompt caching | [`website/docs/developer-guide/context-compression-and-caching.md`](./website/docs/developer-guide/context-compression-and-caching.md) |
| Provider resolution, fallback, or API mode | [`website/docs/developer-guide/provider-runtime.md`](./website/docs/developer-guide/provider-runtime.md) |
| General plugins, hooks, or plugin CLI/slash extensions | [`website/docs/developer-guide/plugins/index.md`](./website/docs/developer-guide/plugins/index.md) |
| Model-provider plugins | [`website/docs/developer-guide/model-provider-plugin.md`](./website/docs/developer-guide/model-provider-plugin.md) |
| Built-in CLI or messaging slash commands | [`website/docs/reference/slash-commands.md`](./website/docs/reference/slash-commands.md) and [`website/docs/developer-guide/gateway-internals.md`](./website/docs/developer-guide/gateway-internals.md) |
| User config, terminal backends, worktree config | [`website/docs/user-guide/configuration.md`](./website/docs/user-guide/configuration.md) |
| Session identity/lifecycle or profile routing | [`docs/session-lifecycle.md`](./docs/session-lifecycle.md) and [`docs/profile-routing.md`](./docs/profile-routing.md) |
| Messaging platform adapter | [`gateway/platforms/ADDING_A_PLATFORM.md`](./gateway/platforms/ADDING_A_PLATFORM.md) |
| Cron scheduler, jobs, delivery, or isolation | [`website/docs/developer-guide/cron-internals.md`](./website/docs/developer-guide/cron-internals.md) |
| Delegation, child budgets, nesting, or durability | [`website/docs/user-guide/features/delegation.md`](./website/docs/user-guide/features/delegation.md) |
| Ink TUI | [`ui-tui/README.md`](./ui-tui/README.md) |
| Dashboard/web UI | [`web/README.md`](./web/README.md) |
| Electron desktop | [`apps/desktop/AGENTS.md`](./apps/desktop/AGENTS.md) and [`apps/desktop/DESIGN.md`](./apps/desktop/DESIGN.md) |
| ZPK package metadata or service restart contract | [`zpk/README.md`](./zpk/README.md) |
| Zettlab device runtime configuration | [Complete Zettlab Engineering Hard Rules](https://github.com/zettlab/zettlab-product-dev/blob/main/.agents/rules/engineering-hard-rules.md) and the owning repository's `zpk/config/<repo>.yaml` |
| Curator behavior | [`website/docs/user-guide/features/curator.md`](./website/docs/user-guide/features/curator.md) |
| Kanban behavior | [`website/docs/user-guide/features/kanban.md`](./website/docs/user-guide/features/kanban.md) |

Before adding a new capability, search source, tests, merged work, and existing
extension points. Prefer extending an existing seam over creating another
framework or parallel implementation.

## Architecture invariants

- `run_agent.py` owns the conversation loop. `model_tools.py` orchestrates tool
  discovery/dispatch; `tools/registry.py` is the low-dependency registry used by
  tool modules. Avoid introducing reverse imports into this chain.
- Tools self-register, but registration does not expose a tool to an agent. A
  core tool must also belong to an intentional toolset in `toolsets.py`.
- Local or third-party capabilities should normally be skills or standalone
  plugins. Add a core tool only when it truly requires harness-owned auth,
  streaming/binary handling, or exact Python integration. New third-party product
  integrations and new memory providers do not land in-tree; widen a generic
  plugin hook when necessary instead of special-casing a vendor in core.
- The OpenAI-compatible chat/SSE contract is a hot path for Zettlab App, Web,
  local-server, and gateway. Unknown optional extensions must remain ignorable by
  old consumers.
- `hermes_cli/commands.py::COMMAND_REGISTRY` is the canonical catalog for
  built-in slash commands. Keep CLI/gateway dispatch, help, messaging menus, and
  autocomplete derived from it; do not create a parallel command list.
- Past conversation context, enabled toolsets, memory snapshots, and cached
  system-prompt layers must remain stable during a session. Cache-affecting slash
  commands should defer changes to the next session unless the user explicitly
  requests immediate invalidation.
- The dashboard chat embeds the Ink TUI through a PTY; it must not reimplement
  the transcript or composer in React. Supporting React panels are allowed when
  they do not become a second chat surface.
- Electron desktop is a separate chat surface backed by `tui_gateway`. Keep
  Electron, renderer, and backend authority boundaries described in its local
  instructions. User skills and quick commands remain executable/discoverable
  even when built-in desktop commands are curated.
- Gateway control commands that must interrupt or answer a running agent must
  bypass both the base-adapter active-session queue and the gateway runner guard.
  Test the real blocked-agent path when adding one.

## Implementation guardrails

<a id="profiles-multi-instance-support"></a>

### Configuration and profiles

- Secrets only belong in `~/.hermes/.env`. Non-secret settings belong in
  `~/.hermes/config.yaml`; device runtime config follows Hard Rule 6.
- Hermes has multiple config-loading paths. Trace the actual caller before
  adding a setting and test every runtime that consumes it; do not assume a CLI
  default is visible to the gateway's direct YAML loader.
- Use `get_hermes_home()` for state paths and `display_hermes_home()` for
  user-facing paths. Never hardcode `~/.hermes` or `Path.home() / ".hermes"` in
  profile-scoped code.
- Profile-management roots are intentionally HOME-anchored so every profile can
  be listed. Do not convert them mechanically to the active `HERMES_HOME`.
- Adapters with unique credentials should acquire and release scoped token locks
  so two profiles cannot use the same credential concurrently.

<a id="adding-new-tools"></a>

### Adding New Tools, plugins, and dependencies

- Tool schemas must not hardcode references to tools from other toolsets. Add
  availability-aware guidance when definitions are assembled instead.
- Persistent tool/plugin state must be profile-scoped. Tests must never write to
  a real `~/.hermes` directory.
- Plugin-specific behavior stays behind generic hooks and registration surfaces;
  core files must not import or branch on a particular third-party plugin.
- PyPI dependencies require a lower bound and an upper ceiling. Git dependencies
  and GitHub Actions require full commit SHAs; CI-only installs use exact pins.
  Regenerate the relevant lockfile after dependency changes.
- Do not add new `simple_term_menu` call sites; use the repository's curses UI.
  Do not use ANSI erase-to-end-of-line in spinner/display code because it leaks
  through `prompt_toolkit`; clear with explicit padding.
- `model_tools._last_resolved_tool_names` is process-global and is temporarily
  saved/restored around delegated children. Treat reads as scoped state, not a
  durable cross-child truth.

### TypeScript

- Use strict TypeScript and avoid `any`.
- Shared renderer state should use small, feature-owned stores; route roots stay
  thin and hooks own one narrow job. Keep persistence beside the owning state.
- Use interfaces for public props/shared object shapes and extend React primitive
  props where possible. Make fire-and-forget async UI handlers explicit with
  `void`.

## Testing and verification

- Behavior changes require both focused unit coverage and a flow/integration/E2E
  proof at the nearest real boundary. Mock external providers at the boundary;
  unit tests must not use live networks or credentials.
- Use `scripts/run_tests.sh`, not raw `pytest`, for authoritative Python test
  runs. It supplies the hermetic credential, HOME, timezone, locale, and
  per-file subprocess isolation via `scripts/run_tests_parallel.py`. Target a
  file and use `-k` for a single test; the runner is file-granular. Run the
  required broader scope before submission.
- Keep `HERMES_HOME` inside a temporary directory in tests. Profile tests that
  exercise HOME-anchored roots must also replace `Path.home()`.
- Test contracts and relationships, not snapshots of expected-to-change data.
  Do not assert model catalog members, config-version literals, enumeration
  counts, or other routine inventory. Assert invariants such as resolution,
  migration to the current version, and catalog metadata completeness.
- Wire live paths with real imports in at least one integration test. Mock-only
  tests do not prove that discovered modules, plugins, tools, or commands are
  actually reachable.
- In the Zettlab monorepo, the repository QA entry point is
  `just qa --only hermes-agent`; follow the parent completion rules for when the
  full harness is required.

Before claiming completion, inspect the task diff, run the smallest sufficient
format/lint/type/test/build set, and report exactly what ran and what did not.
Do not treat a passing unit test as proof of a changed cross-service or UI path.

## Maintaining this file

- Keep `AGENTS.md` below **16 KiB**. If a new section would exceed the budget,
  put the detail in a stable task-specific document and add one routing row here.
- Keep only cross-cutting rules and costly-to-miss invariants. Do not add current
  file counts, exhaustive trees, model/provider/toolset lists, command catalogs,
  config-key inventories, default-value snapshots, or long code examples.
- Do not duplicate user-facing feature documentation. Link to its authoritative
  page and keep at most the invariant needed to prevent a high-risk mistake.
- When behavior changes, update the nearest authoritative doc/test first, then
  adjust this router only if the read path or a broad invariant changed.
