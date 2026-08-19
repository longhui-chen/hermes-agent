# Zettlab Deep Memory Provider

Hermes Memory Provider plugin for the on-device Zettlab Deep Memory MCP
service. Hermes keeps its built-in `MEMORY.md` and `USER.md` memory enabled;
this provider mirrors committed built-in writes and can inject or supplement
authorized recall according to the configured chat mode.

## Lifecycle

- In `always` mode, `on_turn_start()` starts `memo_recall` in a background
  thread and `prefetch()` consumes it before the model API call. If the backend
  is still slow after the bounded hot-path wait, injection is skipped for the
  turn and `memo_recall` remains available as an active-search backstop.
- In `smart` mode, an explicit native `search_memory` call also runs one
  bounded `memo_recall` and combines native and Deep results. There is no
  turn-start prefetch and direct `memo_recall` is not model-facing.
- In `off` mode, chat recall and model-facing Deep tools are disabled.
- `on_memory_write()` durably enqueues successful built-in `add`, `replace`,
  and `remove` mutations in a profile-scoped SQLite outbox. It does not call
  the auxiliary model on the native-memory write path.
- One bounded worker drains the outbox through MCP
  `tools/call(name="memo_write")`. Busy, unavailable, timeout, network, and
  malformed-response failures use bounded exponential backoff and survive
  Hermes or device restarts.
- Every queued mutation keeps one stable MCP call ID. If Hermes stops after
  local-server commits but before the row is acknowledged, local-server's
  candidate-key uniqueness makes replay idempotent.
- Deterministic structured failures are retried three times and then retained
  as a dead-letter row for diagnosis. They are never converted into an
  unstructured or heuristic write.
- The MCP auxiliary model resolves replacement/removal targets from an
  authority-fenced candidate set before revision-checked persistence.
- The MCP service owns model-based extraction, AtomicFact validation, graph
  generation, conflict decisions, scoring, and persistence.

`memo_write` is not exposed as a model tool. There is no heuristic write
fallback: only local-server's validated AtomicFact output can create or revise
a Deep Memory fact. The queue is capped at 512 rows and each native-memory
field at 32,000 characters so a failing backend cannot grow device state
without bound.

## Configuration

Set the provider and managed loopback credentials in the active Hermes profile:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: zettlab_deep_memory
  deep_memory_mode: always  # off | smart | always
```

The default is `always` for compatibility with existing profiles. Mode changes
apply to new chat sessions. All three modes keep the durable `on_memory_write`
mirror active so disabling chat recall does not create a write gap.

```dotenv
ZETTLAB_DEEP_MEMORY_URL=http://127.0.0.1:PORT/api/v1/internal/deep-memory
ZETTLAB_AGENT_ACTION_TOKEN=managed-action-token
```

The URL must be loopback HTTP. Runtime identity is supplied by Hermes metadata,
not accepted from model tool arguments. The queue is stored beneath the active
profile's `hermes_home` as
`zettlab_deep_memory/mirror_outbox.sqlite3`, with directory mode `0700` and
database/sidecar mode `0600`.

The database uses SQLite rollback journaling (`journal_mode=DELETE`) with
`synchronous=FULL`. The target device's SQLite 3.40 runtime is affected by the
upstream WAL-reset corruption issue, and this single-consumer queue does not
need WAL concurrency.
