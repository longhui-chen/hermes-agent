# Zet Agent SSE Extension Contract (v1)

> **Status:** frozen by `chat-ui-b0-contract-freeze` (zettlab-product-dev
> `openspec/changes/chat-ui-b0-contract-freeze`). Changes are additive only;
> every frame listed here has a golden sample in
> `zettlab-product-dev/contracts/chat-ui/v1/golden/hermes/` and the root
> contract guard (`scripts/chat-ui-contract/verify.mjs --scope hermes`)
> fails when a `payload.type` literal in `gateway/platforms/zet_agent.py`
> is missing from this document or from the golden set.

This document is the source of truth for everything the Zettlab adapter
(`gateway/platforms/zet_agent.py`, a subclass of the upstream
`APIServerAdapter`) adds to the `/v1/chat/completions` SSE stream. The
upstream OpenAI-compatible chunks (`chat.completion.chunk` with
`delta.content`) are not part of this contract; local-server consumes them
as `text.delta`.

Consumers: `zettlab-local-server/internal/backend/hermes/translate.go`
(one-to-one translation into `chatproto` events). Nothing else may parse
these frames.

---

## 1. Transport

| SSE event name | Carrier | Who writes it |
|---|---|---|
| `hermes.tool.progress` | one lane for every extension frame, multiplexed by `payload.type`; tool lifecycle frames have **no** `type` and are keyed by `toolCallId` | `api_server.py` `_emit` writer, fed by `("__tool_progress__", payload)` tuples the adapter pushes onto the request's `stream_q` |
| `hermes.error` | error frame emitted once before the finish chunk when the run did not complete cleanly | `api_server.py` (`_chat_stream_error_payload`) |
| finish chunk | the terminal `chat.completion.chunk` with `finish_reason` and a `hermes` extension object | `api_server.py` |

Frames are written with `json.dumps(payload, ensure_ascii=False)` as a
single `data:` line. Field order is not significant.

### 1.1 `turn_id` on extension frames

Every frame the adapter builds carries an optional `turn_id` (the
`metadata.turn_id` local-server sent with the request, read from the
session environment `HERMES_TURN_ID`). It is **present** whenever the
frame is built inside the request's turn context, which includes the
native title push (`_push_title` runs inside the agent run's bound turn)
and every callback the agent loop invokes. It **may be absent** only on
frames built outside a bound turn (compaction rotation triggered outside a
request, background subagent frames after the parent turn ended, which
then carry the turn they were dispatched from). Consumers must not reject a
frame for a missing `turn_id`, and must not use it as a business proof (it
is a correlation label; see zettlab-product-dev total plan §1.4).

The stamp is applied on a shallow copy at the single emit point
(`_put_progress`); adapter-side caches (approval projection, pending
interaction mirrors) never see the wire-only field. Legacy approval frames
set `turn_id` explicitly from the captured owner turn, the same source the
durable path uses.

Tool lifecycle frames (§3) are built by upstream `api_server.py` and carry
no `turn_id`; they are request-scoped by construction.

---

## 2. Semantic frames (`payload.type`)

All frames: `type` (string, required), `turn_id` (string, optional, §1.1).

### `reasoning.delta`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `text` | string | yes | one reasoning token batch |

Producer: `_reasoning_cb` (agent `reasoning_callback`). Translated to
`reasoning.delta`.

### `conversation.title`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `title` | string | yes | LLM-generated session title |

Producer: `_push_title`. local-server adds `session_id` while translating.
`turn_id` is present: the push runs inside the run's bound turn context.

### `context.compaction`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `state` | string | yes | `start` / `running` / `done` / `error` |
| `message` | string | no | human-readable note |
| `old_session_id` | string | no | rotation source |
| `new_session_id` | string | no | rotation target |

Producer: status hook rewriting the upstream compaction status event.

### `steer_dropped`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `text` | string | yes | the steer text that never reached the model |

Producer: turn finalizer when a queued steer was not consumed. Translated to
`steer.dropped{reason: unconsumed}`.

### `hermes.approval`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `approval_id` | string | yes | stable approval identity |
| `interaction_id` | string | durable path | durable interaction id |
| `interaction_generation` | int | durable path | monotonic generation |
| `interaction_delivery_version` | int | durable path | always `1` |
| `command` | string | yes | command awaiting approval |
| `description` | string | yes | human description |
| `pattern_key` | string | yes | primary allow pattern |
| `pattern_keys` | string[] | yes | all patterns |
| `expires_at_ms` | int (unix ms) | yes | deadline |
| `validation_target` | string | no | e.g. `terminal` |

Two shapes exist today: the durable shape (with the three `interaction_*`
fields) and the legacy FIFO shape (without them). Both are pinned in
`golden/hermes/hermes.approval.json` (durable) and documented here; the
legacy shape is retired in `chat-ui-b2a-interactive-protocol`.

### `hermes.clarify`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `clarify_id` | string | yes | equals `interaction_id` |
| `interaction_id` | string | yes | durable interaction id |
| `interaction_generation` | int | yes | monotonic generation |
| `interaction_delivery_version` | int | yes | always `1` |
| `question` | string | yes | question text |
| `choices_offered` | string[] | yes | may be empty (free text) |
| `expires_at_ms` | int (unix ms) | yes | deadline (`CLARIFY_RESPONSE_TIMEOUT`, 300 s today) |
| `turn_id` | string | when known | see §1.1 |

### `hermes.todo`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `todos` | object[] | yes | `{id, content, status, group_index?, plan_id?}` |
| `summary` | object | yes | `{total, pending, in_progress, completed, cancelled}` |

### `hermes.plan`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `plan_id` | string | yes | may be empty on old builds |
| `title` | string | yes | plan title |
| `groups` | object[] | yes | `{icon, label, count, items[]}` |
| `auto_execute` | bool | yes | true = read-only auto card |

Scheduled for removal in `chat-ui-b2b-plan-mode` (present_plan retirement).
Documented here because it is on the wire today.

### `hermes.attachment`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `attachment` | object | yes | `ChatAttachmentWire` (see `schemas/chatproto.schema.json`): `id, category?, kind, v, state, payload, actions?, resolution?, expires_at?, dedup_key?, fallback_text?` |

Producers: `_push_memory_citations`, `_push_memory_saved`, plugin
`ctx.emit_attachment` (creation-governor recommendation cards). `category`
is the semantic UI map class added by this contract; producers may omit it
until the projector (`chat-ui-b2a` / `b3`) fills it.

### `hermes.delegation.progress`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `kind` | string | yes | `delegation` |
| `event` | string | yes | `subagent.start` / `subagent.tool` / `subagent.progress` / `subagent.complete` |
| `tool` | string | no | tool name on `subagent.tool` |
| `preview` | string | no | bounded preview text |
| `task_index`, `task_count` | int | no | batch position |
| `goal`, `subagent_id`, `child_session_id` | string | no | task identity |
| `tool_count` | int | no | running counter |
| `status` | string | no | terminal status on complete |
| `duration_seconds` | number | no | on complete |

---

## 3. Tool lifecycle frames (no `type`)

Built by upstream `api_server.py` (`_on_tool_start` / `_on_tool_complete`
/ `_tool_completion_payload`); keyed by `toolCallId`.

`status: running`

| Field | Type | Req |
|---|---|---|
| `tool` | string | yes |
| `emoji` | string | yes |
| `label` | string | yes |
| `toolCallId` | string | yes |
| `status` | `"running"` | yes |

`status: completed`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `tool`, `toolCallId` | string | yes | |
| `status` | `"completed"` | yes | |
| `outcome` | `"success"` / `"error"` | yes | |
| `output` | object | no | media tools only (`image_generate`, `video_generate`): `success`, `host_image`, `image`, … |
| `ui_hint` | object | no | `takeover_browser` hint `{type, agent_id, browser_session_id, tab_id}`; translated into `tool.result.output.ui_hint` (on the **do-not-touch** list of the chat-ui contract) |
| `browserState` | object | no | bounded page-state projection |
| `browserContentEvidence` | object | no | bounded content projection |
| `error`, `connectorError` | mixed | no | promoted tool errors |

---

## 4. Error frames

### `event: hermes.error`

| Field | Type | Req | Meaning |
|---|---|---|---|
| `message` | string | yes | redacted error text |
| `code` | string | yes | `output_truncated` (length) or provider code or `agent_error` |
| `completed`, `partial`, `failed` | bool | yes | run result flags |
| `reason`, `provider`, `model`, `status_code`, `provider_error_code`, `provider_message` | string / int | no | provider classification |

### Finish chunk

`choices[0].finish_reason` is `stop` / `length` / `error`. When it is not
`stop`, the chunk also carries `error: {message, type}` and
`hermes: {completed, partial, failed, error, error_code}`; `hermes.canonical_final_response`
is present whenever an output transform ran.

Contract: `hermes.error.code` and `finish_chunk.hermes.error_code` MUST be
the same value for the same failure (`_hermes_error_code` is the shared
rule). `chat-ui-b1-lifecycle-error` collapses the two into one emission
path; until then local-server translates `hermes.error` and treats the
finish chunk as confirmation.

---

## 5. Do-not-touch list (from the chat-ui contract)

`metadata.turn_id` (video_edit workflow key), `X-Zettlab-Turn-Id` outside
agent-search, `tool.result.output.ui_hint`, `goal.status`,
`message.appended` outside clarify, `attachment.action.ack`.
