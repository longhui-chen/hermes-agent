# Creation Recommendation Action Receipt Contract

Status: proposed v1 clarification

Owners: Hermes creation-governor and Zettlab Web chat UI

Transport: Zettlab Local Server chat protocol

## Purpose

A creation recommendation card can ask the user to start an Agent, Skill, or
scheduled-task flow, dismiss the proposal, or change the conversation's
recommendation preference. The Web client needs an authoritative answer to one
question: did Hermes accept that card action?

This receipt does not report whether the eventual Agent, Skill, or scheduled
task was created. Resource completion remains the responsibility of the native
Agent Creator, `skill_manage`, or `cronjob` flow.

The protocol has three layers:

1. Chat delivery: did the action message reach Hermes?
2. Recommendation action receipt: did creation-governor accept the action?
3. Resource outcome: did the downstream native flow create the resource?

This contract covers layer 2 only.

## Message identity

An action attempt is identified by:

```text
outer message turn_id + proposal_id + action
```

`turn_id` identifies one click or retry. `proposal_id` identifies the card.
`action` identifies the requested operation. The protocol does not define a
separate `action_id`, and it does not duplicate `turn_id` inside the encoded
payload.

The Local Server transports and persists the outer `turn_id`. For streaming
Hermes responses it also treats Hermes' terminal
`hermes.canonical_final_response` as
the canonical assistant text and uses it for `turn.end.final_text`. This
terminal replacement is required because a streamed model draft cannot retract
a model-authored fake result marker.

## Recommendation capability

Local Server declares that it can replace the streamed draft with Hermes'
canonical terminal response by sending this exact request metadata on every
turn:

```json
{
  "metadata": {
    "creation_action_receipt_transport": "canonical_final_v1"
  }
}
```

The key is request-scoped transport metadata. Hermes consumes it only as
server-side hook context; it is not copied into model messages or provider
request overrides. Missing, non-string, or unknown values are unsupported.

Only a recommendation turn carrying that exact capability may declare that an
action receipt is required:

```json
{
  "version": 1,
  "type": "creation_recommendation",
  "proposal_id": "proposal-123",
  "action_receipts": true
}
```

The capability is per recommendation. A missing or false value identifies a
legacy recommendation for which the Web client retains legacy settlement
behavior. On the OpenAI-compatible `api_server` route, Hermes suppresses the
creation-governor entirely when the transport capability is missing or
unknown, so an old Local Server cannot receive a newly receipt-capable card.

## Action request

The existing v1 request remains unchanged:

```json
{
  "version": 1,
  "type": "creation_recommendation_response",
  "proposal_id": "proposal-123",
  "action": "create",
  "creation_type": "agent",
  "title": "Advertising analyst",
  "dedup_key": "agent:advertising-analyst",
  "evidence_turn_ids": ["source-turn"]
}
```

The request is carried in model content. The Web display marker may also copy
`action_receipts` so paginated history can recover the source card's
capability without loading the source page.

An action against a recommendation with `action_receipts: true` must declare
the same exact request metadata again. This second check is required because a
device may downgrade between displaying a card and sending the click. Hermes
rejects an absent or unknown transport before proposal, preference, or native
creation side effects. Capability on the recommendation turn is not inherited
by a later action turn.

## Action receipt

Hermes appends one trusted hidden result marker to the assistant message for
the same turn:

```json
{
  "version": 1,
  "type": "creation_recommendation_action_result",
  "proposal_id": "proposal-123",
  "action": "create",
  "status": "accepted"
}
```

Rejected results may include one stable, optional `reason_code`:

```json
{
  "version": 1,
  "type": "creation_recommendation_action_result",
  "proposal_id": "proposal-123",
  "action": "mute_session",
  "status": "rejected",
  "reason_code": "preference_not_persisted"
}
```

The initial reason codes are:

| Code | Meaning | Retry policy |
| --- | --- | --- |
| `proposal_not_actionable` | Expired, stale, replayed, wrong-owner, consumed, or field-mismatched proposal | Terminal; do not offer retry |
| `preference_not_persisted` | `mute_session` or `unmute_session` did not commit its durable preference | Retry while the card is actionable |

Unknown or absent reason codes degrade to a generic rejected result. The Web
must not depend on human-readable assistant text to classify a receipt.

The wire form is:

```html
<!--creation-recommendation-action-result BASE64URL_JSON-->
```

For OpenAI-compatible streaming, a transformed response is also carried on the
terminal chunk as a complete server-owned value:

```json
{
  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
  "hermes": {
    "canonical_final_response": "final transformed assistant text"
  }
}
```

The field appears only on a terminal chunk, contains the complete transformed
response rather than a suffix, and coexists with existing Hermes error
metadata. Generic OpenAI clients may ignore the extension. A Local Server that
understands it replaces the accumulated draft before emitting `turn.end`; a
missing field retains legacy accumulation behavior except that Local Server
removes action-result markers from an untrusted streamed draft. This means an
older Hermes cannot accidentally promote a model-authored marker to a receipt.

Hermes includes the terminal field whenever creation-governor authoritatively
emits a receipt or removes a model-authored result marker. This is a semantic
signal from the governor, not a comparison of the pre-hook and post-hook
strings. The field is therefore required even when a model happened to emit the
exact bytes of the governor's canonical receipt and the final text is
byte-identical to the streamed draft. Ordinary replies for which no governor
sanitization or receipt emission occurred omit the extension.

## Action semantics

| Action | `accepted` means | `rejected` means |
| --- | --- | --- |
| `create` | Creation-governor validated the proposal and bound the owning turn to the type-specific native creation route | The proposal was not actionable before Hermes accepted the handoff |
| `dismiss` | The dismissal latch was applied and the proposal was closed | The proposal was not actionable or the latch could not be applied |
| `mute_session` | The muted preference committed durably, or a retry found that same scoped session already durably muted | The proposal was not actionable or persistence failed |
| `unmute_session` | The unmuted preference committed durably, or a retry found that same scoped session already durably unmuted | The request was not actionable or persistence failed |

An accepted `create` can lead to a clarification question, a confirmation step,
or a normal native-flow error explanation. It still does not assert that a
resource exists.

The handoff begins when creation-governor validates the action and binds that
turn to the type-specific native route context. A successful tool call and a
successful chat terminal are not preconditions for `accepted`: the native
Agent, Skill, or Task flow may ask for configuration, fail after a side effect,
or lose its final response. Once accepted, the card therefore stays settled.
This monotonic rule prevents an ambiguous retry from creating a duplicate
resource. A future resource-created result would require a separate idempotent
domain event from the owning native flow.

`unmute_session` is a conversation-level preference operation. Its
`proposal_id` is required for attempt correlation, but it need not identify the
card that originally muted the conversation because every card covered by the
session mute overlay can expose Undo. A state-changing unmute is accepted only
while that scoped session is currently muted. After a committed preference
action loses its receipt, a structurally valid retry for the same scoped
session returns `accepted` without another write when the durable target value
is already present. An absent preference row is not proof of a prior unmute and
does not qualify for this reconciliation. The retry still requires a non-empty
outer `turn_id` and `proposal_id`, and the receipt echoes the retry's exact
`proposal_id + action` under that exact turn.

## Authority and precedence

A valid receipt is the domain result for the card action and is more specific
than the assistant message's generic terminal status.

The Web resolves one attempt in this order:

1. `queued` or `sending` user action: pending.
2. Exact `turn_id + proposal_id + action` receipt:
   - accepted: settled;
   - rejected: failed or unavailable according to `reason_code`.
3. No receipt and an exact-turn terminal:
   - `create`: uncertain and non-retryable from the card, because Hermes may
     already have committed a downstream side effect;
   - dismiss or preference actions: failed and retryable when still actionable.
4. History may still have unloaded newer messages: unconfirmed, not failed.
5. A legacy recommendation without receipt capability settles after successful
   delivery according to the legacy contract.

This precedence is required because handoff, dismiss, and preference changes
commit in the pre-hook. A later narration failure cannot roll back a committed
domain action. For create, a generic terminal can never revoke an accepted
handoff.

Conflicting accepted and rejected receipts for the same attempt are protocol
corruption and must fail closed.

## Producer invariants

- A valid new Web action has a non-empty outer `turn_id`.
- A receipt-capable recommendation and every action against it independently
  declare `creation_action_receipt_transport: canonical_final_v1`.
- Missing or unknown transport capability never enables receipt production;
  a downgrade between card delivery and action fails before action mutation.
- Capability is isolated to one API request and supplied identically to the
  governor pre-hook and transform-hook. It never leaks to the next turn or to
  the model/provider request.
- Hermes stores a pending receipt with that exact `turn_id`.
- The finalizer consumes only the pending receipt for the same turn.
- One action turn emits at most one trusted receipt.
- Hermes strips model-authored result markers before appending its own marker
  to the canonical final response.
- Governor receipt emission or marker sanitization requires canonical terminal
  delivery even when the canonical response is byte-identical to the model
  draft. A normal untransformed reply does not require canonical delivery.
- The streaming API carries that canonical transformed response in the
  terminal `hermes.canonical_final_response` field. Local Server replaces its
  accumulated draft with that value before emitting and persisting
  `turn.end.final_text`.
- Canonical provenance is enforced at the Hermes-to-Local trust boundary. Once
  Local Server has replaced the draft, the canonical assistant text and outer
  `turn_id` are the history source of truth; Web and persisted history do not
  require a new provenance field for receipt correctness.
- Web derives action results only from canonical terminal/history content, not
  from an in-flight streamed draft.
- A pending receipt is cleared after the owning turn ends and never leaks to a
  later turn.
- Invalid, stale, replayed, wrong-owner, or mismatched requests do not mutate a
  current valid proposal.
- Accepted create closes the proposal and is never reopened by a generic turn
  failure. A missing create receipt after dispatch is uncertain, not retryable.
- A message containing a recommendation-response wrapper that fails v1
  validation is rejected as protocol input and never falls through to legacy
  natural-language action matching.

## Projection invariants

- A later rejected duplicate create cannot reopen an already accepted create.
- A failed mute action on another card cannot cover a settled proposal card.
- Session mute state changes only on accepted mute or unmute receipts.
- A rejected unmute leaves the last accepted mute state intact.
- An expired, unavailable, or terminally rejected card never renders an
  executable retry control.
- Refresh and pagination preserve the same result as the live stream.

## Legacy history

- `action_receipts` missing or false: preserve legacy delivery-based behavior.
- Existing v1 result markers remain valid; no data migration is required.
- History without `turn_id` may use only a bounded contiguous legacy segment,
  matching both `proposal_id` and `action` and stopping at the next user action
  boundary.
- If newer history is not fully loaded, absence of a receipt is not proof of
  rejection.
- Paginated Web messages use the Local Server's stable database row `id`, not a
  page index.

## Deployment order

The request capability is the executable Local-first version fence. A document
or deployment intention alone is not a fence: an old Local Server ignores the
Hermes terminal extension and can pass a model-authored result marker through
as ordinary assistant text. Persisting canonical provenance in Web or history
would not repair that old hop.

Rollout order is:

1. Deploy Web compatibility for both legacy and receipt-capable cards.
2. Deploy Local Server canonical replacement and untrusted-marker removal.
3. Make upgraded Local Server send
   `creation_action_receipt_transport: canonical_final_v1` on both
   recommendation and action turns.
4. Deploy Hermes receipt production. Hermes emits `action_receipts: true` and
   performs receipt-capable actions only for requests carrying the exact
   capability; missing and future/unknown values fail closed.

This negotiation safely supports a mixed Local fleet: old instances never opt
in, and a device that rolls back after displaying a capable card cannot mutate
through the receipt path. It is not replaced by carrying provenance to Web. No
broad Web/history provenance schema is required for this v1 contract. History
that was already accepted through an unsafe pre-fence deployment, if any, must
be quarantined or migrated separately; enabling the fence cannot retroactively
establish its provenance.
