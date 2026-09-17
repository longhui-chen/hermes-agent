# Context Compression and Caching

Hermes Agent uses a dual compression system and Anthropic prompt caching to
manage context window usage efficiently across long conversations.

Source files: `agent/context_engine.py` (ABC), `agent/context_compressor.py` (default engine),
`agent/prompt_caching.py`, `gateway/run.py` (session hygiene), `run_agent.py` (search for `_compress_context`)


## Pluggable Context Engine

Context management is built on the `ContextEngine` ABC (`agent/context_engine.py`). The built-in `ContextCompressor` is the default implementation, but plugins can replace it with alternative engines (e.g., Lossless Context Management).

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
  engine: "lcm"           # example — plugin providing lossless context
```

The engine is responsible for:
- Deciding when compaction should fire (`should_compress()`)
- Performing compaction (`compress()`)
- Optionally exposing tools the agent can call (e.g., `lcm_grep`)
- Tracking token usage from API responses

Selection is config-driven via `context.engine` in `config.yaml`. The resolution order:
1. Check `plugins/context_engine/<name>/` directory
2. Check general plugin system (`register_context_engine()`)
3. Fall back to built-in `ContextCompressor`

Plugin engines are **never auto-activated** — the user must explicitly set `context.engine` to the plugin's name. The default `"compressor"` always uses the built-in.

Configure via `hermes plugins` → Provider Plugins → Context Engine, or edit `config.yaml` directly.

For building a context engine plugin, see [Context Engine Plugins](/developer-guide/context-engine-plugin).

## Dual Compression System

Hermes has two separate compression layers that operate independently:

```
                     ┌──────────────────────────┐
  Incoming message   │   Gateway Session Hygiene │  Safety check at 90% of context
  ─────────────────► │   (pre-agent, API usage) │  Safety net for large sessions
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Agent ContextCompressor │  90% of window, capped at 244,800 (default)
                     │   (in-loop, real tokens)  │  Normal context management
                     └──────────────────────────┘
```

### 1. Gateway Session Hygiene (90% safety check)

Located in `gateway/run.py` (search for `Session hygiene: auto-compress`). This is a **safety net** that
runs before the agent processes a message. It prevents API failures when sessions
grow too large between turns (e.g., overnight accumulation in Telegram/Discord).

- **Threshold**: 90% of model context length, using provider-reported usage only. Without usage, ordinary pressure is deferred to the next provider response; the existing extreme-message-count guard remains.
- **Token source**: Actual API-reported tokens from the last turn. Rough stored-history estimates are diagnostic only; the agent checks projected request pressure.
- **Fires**: Only when `len(history) >= 4` and compression is enabled
- **Purpose**: Catch sessions that escaped the agent's own compressor

The gateway safety threshold cannot precede the default agent trigger. Raw stored
reasoning is not a valid reason to summarize before request projection.

### 2. Agent ContextCompressor (Codex-style standard budget)

Located in `agent/context_compressor.py`. This is the **primary compression
system** that runs inside the agent's tool loop with access to accurate,
API-reported token counts.


## Configuration

All compression settings are read from `config.yaml` under the `compression` key:

```yaml
compression:
  enabled: true              # Enable/disable compression (default: true)
  threshold: 0.90            # Fraction of route window, capped at 90%
  threshold_tokens: 244800   # Standard working budget: 272000 * 90%
  # model_thresholds:        # Per-model threshold overrides (substring match,
  #   "glm-5.2": 0.40        # longest key wins). See "Per-model threshold
  #   "claude-sonnet": 0.35  # overrides" below.
  target_ratio: 0.20         # How much of threshold to keep as tail (default: 0.20)
  protect_last_n: 20         # Minimum protected tail messages (default: 20)
  min_tail_user_messages: 1  # Real user messages guaranteed in the tail (default: 1)
  codex_gpt55_autoraise: true  # gpt-5.5 on Codex OAuth: raise trigger to 85% (default: true)
  codex_gpt55_autoraise_notice: true  # Show the one-time autoraise notice (default: true)
  codex_app_server_auto: native  # native|hermes|off for Codex app-server thread compaction
  in_place: true             # Compact on the same session id, no rotation (default: true)

# Summarization model/provider configured under auxiliary:
auxiliary:
  compression:
    model: null              # Override model for summaries (default: auto-detect)
    provider: auto           # Provider: "auto", "openrouter", "nous", "main", etc.
    base_url: null           # Custom OpenAI-compatible endpoint
```

### Parameter Details

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `threshold` | `0.90` | >0, effective maximum 0.90 | Ratio of route window; no hidden 75% or 64k lower bound |
| `threshold_tokens` | `244800` | positive integer or null | Absolute working-context trigger cap; explicit null disables this cap for a verified larger route |
| `model_thresholds` | `{}` | map | Per-model overrides of `threshold`. Keys are substring-matched against the model name (longest match wins). The 90% ceiling and absolute cap apply on top |
| `target_ratio` | `0.20` | 0.10-0.80 | Controls tail protection token budget: `threshold_tokens × target_ratio` |
| `protect_last_n` | `20` | ≥1 | Minimum number of recent messages always preserved |
| `min_tail_user_messages` | `1` | ≥1 | Minimum number of REAL (actionable) user messages guaranteed to survive in the uncompressed tail. `1` = the existing single last-user anchor (behavior-preserving default). Raise to e.g. `3` to keep the last 3 real user turns verbatim even when bulky tool outputs fill the tail token budget. Blank platform echoes, compaction handoffs, and synthetic continuation rows never count toward N. The guarantee wins over the tail token budget — the tail may exceed the budget when the anchor pulls the cut back |
| `protect_first_n` | `3` | (hardcoded) | System prompt + first exchange always preserved |
| `idle_compact_after_seconds` | `0` | ≥0 seconds | Opt-in: compact up front when a session resumes after this many seconds idle (0 = disabled). Skips when context ≤ threshold × target_ratio; honors cooldown/anti-thrash/lock guards |
| `codex_gpt55_autoraise` | `true` | bool | Raise the trigger to 85% for gpt-5.5 on the ChatGPT Codex OAuth route (see below). Set `false` to keep the global `threshold` |
| `codex_gpt55_autoraise_notice` | `true` | bool | Show the one-time Codex gpt-5.5 autoraise notice. Set `false` to keep the 85% autoraise but suppress the banner |
| `codex_app_server_auto` | `native` | `native`, `hermes`, `off` | Thread-compaction mode for Codex app-server sessions (see below) |
| `in_place` | `true` | bool | Compact on the same session id instead of rotating to a new one (see below) |

### In-place compaction (single stable session id)

With `compression.in_place: true` (the default), a compaction **rewrites the live message list on the same session id**: the system prompt is rebuilt, the summarized middle is swapped in, and the pre-compaction turns are soft-archived under the same id (`active=0, compacted=1` in the session store) — still searchable via `session_search` and recoverable, never deleted. There is no `parent_session_id` chain and no `name #N` renumbering; one conversation keeps one durable id for its whole life. This eliminated the session-rotation bug cluster (lost `/goal` state, orphaned sessions, search gaps across boundaries).

Consumers observe the mode rather than diffing session ids:

- The `session:compress` event carries `in_place: true/false` and `old_session_id` (empty string in in-place mode, since there is no old id).
- The gateway re-baselines transcript handling from the agent's rotation-independent `_last_compaction_in_place` flag, not from an id-change diff.

Set `in_place: false` to restore the legacy rotating path, where each compaction commits a new session id linked to the previous one via `parent_session_id`.

### Per-model threshold overrides

`compression.model_thresholds` lets you trigger compaction at different points
depending on the active model — useful when you swap between models with very
different context windows (e.g. a 1M-context model can compress later while a
128K model should compress earlier):

```yaml
compression:
  threshold: 0.50
  model_thresholds:
    "glm-5.2": 0.40
    "glm-5.2-1M": 0.25
    "claude-sonnet": 0.35
```

Resolution rules:

- Keys are **substring-matched** against the model name; the **longest
  matching key wins** (`glm-5.2-1M` beats `glm-5.2` for model `glm-5.2-1M`).
- When no key matches (or the map is empty), the global `threshold` applies.
- The override is re-resolved on every `/model` switch; switching to a model
  with no matching key falls back to the global `threshold`.
- Lower explicit ratios are honored; the 90% ceiling and absolute cap still apply.

Plugin context engines can reuse the same resolution logic via
`from agent.context_compressor import resolve_model_threshold`; engines that
override `update_model()` own their own compaction policy and may ignore the
map.

### Codex gpt-5.5 threshold autoraise

The ChatGPT Codex OAuth backend hard-caps gpt-5.5 at a **272K** context window
(the same slug exposes 1.05M on OpenAI's direct API and OpenRouter, and 400K on
GitHub Copilot). At the previous 50% default, compaction would fire at ~136K —
half the window the model can actually use. When the active route is Codex
OAuth (`provider: openai-codex`) and the model is gpt-5.5, Hermes raises the
trigger to **85%** (~231K) and shows a notice with the opt-out command. The
notice is shown once per profile — a marker under `$HERMES_HOME`
(`.codex_gpt55_autoraise_notice`) records that it ran, so repeated agent/session
inits (e.g. every inbound gateway message) don't re-emit it; if the raised
threshold later changes it re-notifies once. Only this exact route is affected;
gpt-5.5 on any other provider keeps your global `threshold`. To opt back down to
the global value:

```bash
hermes config set compression.codex_gpt55_autoraise false
```

To keep the 85% autoraise but hide only the one-time notice:

```bash
hermes config set compression.codex_gpt55_autoraise_notice false
```

### Codex app-server thread compaction

Codex app-server sessions (`api_mode: codex_app_server` — the codex CLI/agent
runtime) are different from every other route: the codex agent owns the backing
thread context, so Hermes' auxiliary summarizer cannot shrink it — rewriting the
local transcript mirror leaves the real thread growing unbounded until a hard
context reset. For this runtime, compaction goes through the app-server's own
mechanism instead:

- Manual compaction (`/compress`) asks the app-server to compact the thread
  (`thread/compact/start`) and waits for the compaction turn to complete.
- Automatic compaction is controlled by `compression.codex_app_server_auto`:
  the default `native` lets the app-server decide when to compact and Hermes
  records the resulting compaction events (compression counters, session
  events). Set `hermes` to let Hermes' compression threshold initiate
  app-server compaction, or `off` to disable Hermes-initiated automatic
  compaction entirely (codex may still compact natively).

Hermes' local transcript is never rewritten on this runtime — state.db records
the compaction boundary while the visible transcript stays intact. All other
routes (including Codex OAuth chat sessions) keep Hermes' summary compressor.

### Codex reference and standard working budget

Reference: OpenAI Codex 0.153.4, `codex-rs/protocol/src/openai_models.rs`,
`ModelInfo::auto_compact_token_limit` and the test
`model_context_window_limits_preserve_their_distinct_meanings`.
That test distinguishes a 272,000 window, 258,400 usable context (95%), and
244,800 auto-compaction limit (90%). The usable-context number is NOT a second
compaction trigger. No 95% UI scaling is added to Hermes in this patch.

Hermes reuses its existing absolute-cap setting, rather than adding another
mode/state machine:

```text
route window 272,000 -> 90% = 244,800 -> standard cap 244,800
route window 1,050,000 -> 90% = 945,000 -> standard cap 244,800
route window 128,000 -> 90% = 115,200 -> standard cap does not enlarge it
```

The standard budget is a client policy, NOT proof that a cloud alias supports
272k or 1M. Route metadata/explicit overrides remain the source of capacity;
the existing unknown-model fallback is still an unverified estimate. A caller's
explicit session window now survives a different default route in config.

An explicit chat-provider `max_tokens` remains a separate hard input safety
bound: `min(window * ratio, window - max_tokens, threshold_tokens)`. Unlike the
previous formula, output space is not subtracted before applying the ratio.
This provider adaptation is not claimed to be Codex's formula. Unspecified
output capacity remains unknown; no fabricated 128k or other output reserve.
There is no automatic 64k trigger floor or 75% ratio uplift.

Standard configuration (with route capacity configured separately):

```yaml
compression:
  threshold: 0.90
  threshold_tokens: 244800
```

Explicit long-context configuration, only for a verified route:

```yaml
model:
  context_length: 1000000
compression:
  threshold: 0.90
  threshold_tokens: 900000
```

Existing explicit YAML is not silently rewritten. A deployed old `threshold: 0.5`
now really means 50%; it must be migrated to the standard configuration above
when deploying this fix. An old explicit `threshold_tokens: null` opts out of the
standard cap and must also be reviewed. Device config remains owned by the
packaging repository's `zpk/config/<repo>.yaml`. No shared device was changed.

### Which fixes come from Codex, and which remain Hermes adaptations

- **Thresholds:** port the 90% upper bound and separate explicit compaction cap;
  do not confuse full capacity, usable context and the trigger.
- **Accounting:** Codex `context_manager/history.rs::get_total_token_usage` uses
  the latest provider usage plus subsequent items, and avoids counting reasoning
  twice when already included by the server. Hermes uses successful input usage
  as the automatic trigger, without promoting character estimates to measurements; this is
  an adaptation to Chat Completions, not a claim of identical Responses semantics.
- **Reasoning:** preserve provider-required protocol state. Display summaries are
  not generic conversation content. No blanket deletion of encrypted/native state.
- **Compaction continuity:** preserve active user intent and continuation. Hermes'
  textual auxiliary summaries differ from Codex remote encrypted compaction.
- **No-gain rejection:** retain Hermes' existing failure protection and reject an
  expanded automatic candidate. This is a Hermes safety fix, not attributed to Codex.

## Compression Algorithm

The `ContextCompressor.compress()` method follows a 4-phase algorithm:

### Phase 1: Prune Old Tool Results (cheap, no LLM call)

Old tool results (>200 chars) outside the protected tail are replaced with:
```
[Old tool output cleared to save context space]
```

This is a cheap pre-pass that saves significant tokens from verbose tool
outputs (file contents, terminal output, search results).

### Phase 2: Determine Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│  Message list                                               │
│                                                             │
│  [0..2]  ← protect_first_n (system + first exchange)        │
│  [3..N]  ← middle turns → SUMMARIZED                        │
│  [N..end] ← tail (by token budget OR protect_last_n)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Tail protection is **token-budget based**: walks backward from the end,
accumulating tokens until the budget is exhausted. Falls back to the fixed
`protect_last_n` count if the budget would protect fewer messages.

Boundaries are aligned to avoid splitting tool_call/tool_result groups.
The `_align_boundary_backward()` method walks past consecutive tool results
to find the parent assistant message, keeping groups intact.

### Phase 3: Generate Structured Summary

:::warning Summary model context length
The summary model must have a context window **at least as large** as the main agent model's. The entire middle section is sent to the summary model in a single `call_llm(task="compression")` call. If the summary model's context is smaller, the API returns a context-length error — `_generate_summary()` catches it, logs a warning, and returns `None`. The compressor then drops the middle turns **without a summary**, silently losing conversation context. This is the most common cause of degraded compaction quality.
:::

The middle turns are summarized using the auxiliary LLM with a structured
template:

```
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — specific file paths, commands run, results]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, configuration details]
```

Summary budget scales with the amount of content being compressed:
- Formula: `content_tokens × 0.20` (the `_SUMMARY_RATIO` constant)
- Minimum: 2,000 tokens
- Maximum: `min(context_length × 0.05, 12,000)` tokens

### Phase 4: Assemble Compressed Messages

The compressed message list is:
1. Head messages (with a note appended to system prompt on first compression)
2. Summary message (role chosen to avoid consecutive same-role violations)
3. Tail messages (unmodified)

Orphaned tool_call/tool_result pairs are cleaned up by `_sanitize_tool_pairs()`:
- Tool results referencing removed calls → removed
- Tool calls whose results were removed → stub result injected

### Iterative Re-compression

On subsequent compressions, the previous summary is passed to the LLM with
instructions to **update** it rather than summarize from scratch. This preserves
information across multiple compactions — items move from "In Progress" to "Done",
new progress is added, and obsolete information is removed.

The `_previous_summary` field on the compressor instance stores the last summary
text for this purpose.


## Before/After Example

### Before Compression (45 messages, ~95K tokens)

```
[0] system:    "You are a helpful assistant..." (system prompt)
[1] user:      "Help me set up a FastAPI project"
[2] assistant: <tool_call> terminal: mkdir project </tool_call>
[3] tool:      "directory created"
[4] assistant: <tool_call> write_file: main.py </tool_call>
[5] tool:      "file written (2.3KB)"
    ... 30 more turns of file editing, testing, debugging ...
[38] assistant: <tool_call> terminal: pytest </tool_call>
[39] tool:      "8 passed, 2 failed\n..."  (5KB output)
[40] user:      "Fix the failing tests"
[41] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[42] tool:      "import pytest\n..."  (3KB)
[43] assistant: "I see the issue with the test fixtures..."
[44] user:      "Great, also add error handling"
```

### After Compression (25 messages, ~45K tokens)

```
[0] system:    "You are a helpful assistant...
               [Note: Some earlier conversation turns have been compacted...]"
[1] user:      "Help me set up a FastAPI project"
[2] assistant: "[CONTEXT COMPACTION] Earlier turns were compacted...

               ## Goal
               Set up a FastAPI project with tests and error handling

               ## Progress
               ### Done
               - Created project structure: main.py, tests/, requirements.txt
               - Implemented 5 API endpoints in main.py
               - Wrote 10 test cases in tests/test_api.py
               - 8/10 tests passing

               ### In Progress
               - Fixing 2 failing tests (test_create_user, test_delete_user)

               ## Relevant Files
               - main.py — FastAPI app with 5 endpoints
               - tests/test_api.py — 10 test cases
               - requirements.txt — fastapi, pytest, httpx

               ## Next Steps
               - Fix failing test fixtures
               - Add error handling"
[3] user:      "Fix the failing tests"
[4] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[5] tool:      "import pytest\n..."
[6] assistant: "I see the issue with the test fixtures..."
[7] user:      "Great, also add error handling"
```


## Prompt Caching (Anthropic)

Source: `agent/prompt_caching.py`

Reduces input token costs by ~75% on multi-turn conversations by caching the
conversation prefix. Uses Anthropic's `cache_control` breakpoints.

### Strategy: system_and_3

Anthropic allows a maximum of 4 `cache_control` breakpoints per request. Hermes
uses the "system_and_3" strategy:

```
Breakpoint 1: System prompt           (stable across all turns)
Breakpoint 2: 3rd-to-last non-system message  ─┐
Breakpoint 3: 2nd-to-last non-system message   ├─ Rolling window
Breakpoint 4: Last non-system message          ─┘
```

### How It Works

`apply_anthropic_cache_control()` deep-copies the messages and injects
`cache_control` markers:

```python
# Cache marker format
marker = {"type": "ephemeral"}
# Or for 1-hour TTL:
marker = {"type": "ephemeral", "ttl": "1h"}
```

The marker is applied differently based on content type:

| Content Type | Where Marker Goes |
|-------------|-------------------|
| String content | Converted to `[{"type": "text", "text": ..., "cache_control": ...}]` |
| List content | Added to the last element's dict |
| None/empty | Added as `msg["cache_control"]` |
| Tool messages | Added as `msg["cache_control"]` (native Anthropic only) |

### Cache-Aware Design Patterns

1. **Stable system prompt**: The system prompt is breakpoint 1 and cached across
   all turns. Avoid mutating it mid-conversation (compression appends a note
   only on the first compaction).

2. **Message ordering matters**: Cache hits require prefix matching. Adding or
   removing messages in the middle invalidates the cache for everything after.

3. **Compression cache interaction**: After compression, the cache is invalidated
   for the compressed region but the system prompt cache survives. The rolling
   3-message window re-establishes caching within 1-2 turns.

4. **TTL selection**: Default is `5m` (5 minutes). Use `1h` for long-running
   sessions where the user takes breaks between turns.

5. **Model identity is part of the cache key**: Provider-side caches are scoped
   to the model (and account/API key) serving the request. Any mid-conversation
   model change — an explicit `/model` switch, primary-model fallback, or a
   credential-pool rotation onto a different account — means the next request
   gets zero cache hits and re-reads the full conversation at undiscounted
   input price. This is inherent to how provider caches work, not something
   Hermes can avoid; user-facing docs for `/model`, fallback providers, and
   credential pools carry cost warnings for this reason. Don't add features
   that silently swap the model or credentials mid-session.

### Enabling Prompt Caching

Prompt caching is automatically enabled when:
- The model is an Anthropic Claude model (detected by model name)
- The provider supports `cache_control` (native Anthropic API or OpenRouter)

```yaml
# config.yaml — TTL is configurable (must be "5m" or "1h")
prompt_caching:
  cache_ttl: "5m"
```

The CLI shows caching status at startup:
```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```


## Context Pressure Warnings

Intermediate context-pressure warnings have been removed (see the iteration-budget block in `run_agent.py`, which notes: "No intermediate pressure warnings — they caused models to 'give up' prematurely on complex tasks"). Compression fires when prompt tokens reach the configured `compression.threshold` (default 90% of the window, capped at 244,800 tokens) with no prior warning step; gateway session hygiene fires as the secondary safety net at 90% of the model's context window using reported usage.

### Reasoning accounting and task continuity

Raw storage may contain both `reasoning` (display/trajectory) and
`reasoning_content` (provider replay). Identical traces count once in the shared
estimator and tail budget. Turn-start and post-tool pressure checks project
reasoning through the same replay policy as request construction: display-only
traces do not trigger compaction, while required provider traces are preserved.
This projection does not modify the stored transcript, change provider policy,
or disable thinking. Other transport-specific envelopes remain conservative
estimates; this is not a universal tokenizer or full wire serialization.

Automatic batch compaction with a supplied pressure reading rejects candidates
whose estimated message size is no smaller than the original. It restores the
prior summary state and uses the existing ineffective-compaction breaker rather
than adding another retry loop. Explicit forced compaction can still rebuild a
checkpoint. The provider-overflow recovery path remains available.

A checkpoint must retain the original goal, constraints, evidence of completed
work, and unfinished work. Compaction itself neither completes nor cancels a
task. Later user cancellations and corrections take precedence; speculation
and unverified assistant claims must not become verified accomplishments.

### Provider usage calibration and remaining rollout checks

Successful ordinary requests now attach their preflight estimate to canonical usage.
The existing deferral guard uses that provider-confirmed anchor even before the
first compaction. Pressure checks never advance the anchor. Deferral remains
bounded by the existing growth allowance and stops when observed input plus
estimated growth reaches the threshold. This is not an exact tokenizer and does
not claim that an unknown model alias has a specific tokenizer or context limit.

Validation on the isolated local runtime uses the real configured cloud API,
not an AC owner's device. The API returned model `gpt-5.6-sol`; this is response
metadata, not independent verification of the provider's backend weights.

| Check | Result |
|---|---|
| Three sequential synthetic observation tools | Steps 1/2/3 completed; original GOAL-480 and no-write constraint retained |
| First request count | Provider input 11,326; rough input 14,790 (about 31% high) |
| False-pressure boundary | After the first observed usage, a test-only 14k trigger was installed; following estimates 14,886–15,078 exceeded it while input stayed 11,379–11,484; no compaction |
| Real auxiliary compaction then continuation | Estimated history 38,112 → 4,145; only unfinished steps 2/3 executed; original goal retained |

These are synthetic read-only tool fixtures with real model and summarizer
responses. They prove the local agent loop, not the Memo UI or device release.
They do not substantiate a tenfold system-prompt overestimate. No new exact
model tokenizer is claimed for unknown cloud aliases: successful provider input
usage calibrates the existing bounded guard, while cold starts retain estimates.

Native reasoning fields and inline think blocks are excluded from summary input
by the existing serializer (now explicitly regression-tested). Required replay
fields survive normal API calls; non-required fields are omitted by the existing
provider policy. Tail selection is still conservative for protocol-required
reasoning; this PR does not globally erase persisted thinking or opaque provider
state. All automatic savings comparisons use the host's replay-aware estimator
when supplied; older context engines remain compatible through optional kwargs.

Release checks, separate from the code-review gate: confirm the intended cloud
route's capacity/output defaults and the packaged YAML, then use a dedicated AC
for ARM/RSS/restart and Memo UI acceptance. 32.98 and 35.28 belong to other users
and were not modified or used for this live inference validation.

Engineering constraints: no new process, dependency, permanent payload cache or
business state; projection uses O(message count) transient shallow dictionaries.
No tool permissions, external message contracts or device YAML sources change.
Failure retains the original transcript and existing recovery/cooldown machinery.
Reliability takes precedence over reducing compaction count. The core overlay is
explicitly marked; its size exception requires PR review rather than modifying
the overlay gate. Microsoft/Azure dependency freeze is unaffected.

### Latest standard-budget validation

The Codex-aligned follow-up passes 515 tests across 53 files, including the
272k/244800 boundary, explicit runtime pin, model switching, 1000 randomized
window/output/cap transitions, and the gateway no-usage path. Five unrelated
failures reproduce on unchanged main (telemetry: 2; context selection: 3).

The latest live attempt reports window=272000 and threshold=244800, fixing
the earlier discarded 256k pin. Its model calls failed with cloud overload;
a separate minimal hello request timed out. No latest live tool-flow success
is claimed. Earlier live evidence above belongs to the previous budget revision.
Cloud validation is deferred at the user's direction; no shared AC was modified.

### Provider-measured automatic pressure (PR 480 correction)

Operating envelope: the provider returns valid usage for a completed request.
Chat Completions `prompt_tokens` includes cached input. Output/reasoning usage
is billing information, not an extra input charge. No local tokenizer is added.

| State | Automatic pressure | Behavior |
|---|---|---|
| Latest valid input usage | That request's input total | Compact at configured trigger |
| Unsent tool output / user input | Not measured yet | Submit normally, replace baseline from response |
| No usage / cold start / model change | Unknown (zero trigger signal) | Do not compact from character count |
| Just compacted | Old usage invalid | Wait for new request usage |
| Explicit provider context overflow / HTTP 413 | Existing bounded recovery | Compact/retry or return failure preserving history |
| Generic 400 / timeout plus large rough count | Not evidence of overflow | Existing error handling, no guessed compaction |

This intentionally differs from Codex's estimated-new-items accounting. A sudden
large tool result can cause one rejected request before bounded recovery. It does
not justify repeatedly summarizing fitting contexts. Missing usage must not become
a fabricated precise count. Legacy estimates remain only in upstream diagnostics, auxiliary summary output
budget sizing and optional plugin/micro-compaction paths, never as automatic
capacity evidence in the built-in batch-compaction path. Manual `/compress`
and optional plugin engines retain their existing contracts.

The runtime keeps only existing scalar usage state (no additional resident cache,
model, or tokenizer). Protocol, auth scope, device YAML ownership and dependencies
are unchanged. Reliability tradeoff: bounded explicit-overflow recovery instead of
proactive destructive guesses. Randomized state transitions and a real agent loop
with the provider boundary mocked must validate these invariants.

### Removal of character-derived capacity decisions

Operating envelope: only provider usage measures model input occupancy. Missing
usage remains unknown; byte counts measure payload size, not model capacity.

| Consumer | Allowed signal | Unknown / overflow behavior |
|---|---|---|
| Active and optional idle compaction | Latest provider input usage | Wait for usage, bounded explicit-overflow recovery |
| Reference-model advisory request | Original advisory transcript | Provider rejection becomes failed advice, never guessed history removal |
| File/reference injection | Explicit UTF-8 payload byte cap | Reject oversize attachment with path preserved, independent of route window |
| Capacity display | Provider usage | Explicit unmeasured status; component sizes in bytes |
| Model-switch warning | Previous route's measured input | State that the new route must measure again, no guaranteed compaction claim |

The attachment byte envelope is 1 MiB total and at most 16 references per message.
These are device resource limits, not inferred token limits. Expansion is sequential
and stops at the total envelope; file reads are bounded before decoding. Existing
owner/path/credential restrictions and tool-output resource limits remain intact.
The change adds no tokenizer, resident model, network count call, or new history.
Old optional token-attribution fields stay zero/unknown for wire compatibility.

Built-in batch compaction compares the replay-projected UTF-8 size before/after,
including same-message-count overflow recovery. Recent-tail partitioning uses the
existing summary target ratio of that payload size, preserving tool-pair and latest
user-message boundaries. It does not compare bytes with the model token window.
The old estimated-token feasibility shortcut cannot skip summarization and drop
history on this path. Final compaction effectiveness still awaits provider usage.
Upstream auxiliary summary output-budget sizing and experimental micro compaction
are separate policies; this change does not claim to remove every estimator from
the repository. Old calibration fields and normal-request/post-tool scans are gone.

### 2026-09-16 cloud recheck after character-estimate cleanup

Tested commit `6c67919a6a` against the configured `gpt-5.6-sol` cloud route.
All prompts and tool results were synthetic; no shared AC or user session was modified.

| Probe | Result | Elapsed |
|---|---|---|
| Non-streaming hello | HTTP 200, `HELLO_OK`, input 12 / output 7 tokens | 24.05 s |
| Streaming hello with usage requested | HTTP 200 headers, then SSE `upstream_error`: servers overloaded; no content or usage | 20.54 s |
| Real Agent three-step read-only tool flow | Upstream overload before any tool call; no usage, no compaction | 43.74 s |
| Minimal forced tool call, non-streaming | Connection terminated (`RemoteProtocolError`), no complete response | 30.53 s |

The standard instance resolved window 272000 / trigger 244800. Only basic hello
succeeded: cloud service is not yet stable enough for tool-flow or compaction
continuation acceptance. HTTP 200 alone is not a successful streaming inference.
The planned large tool payload was never produced, so this run does not validate
large-context capacity or successful post-compaction continuation. No compression
implementation change is justified by these upstream failures; preserve the
existing local regression evidence and retry live acceptance after cloud recovery.
