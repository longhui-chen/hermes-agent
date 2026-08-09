# Creation Recommendation Action Receipt Implementation Plan

## Goal

Make creation recommendation cards reflect whether Hermes accepted each card
action, without claiming that a downstream resource was created and without
adding a second attempt identifier.

The contract is defined in
`docs/creation-recommendation-action-receipt-contract.md` and must remain the
source of truth for implementation and review.

## Scope

### Hermes

- Restore v1 action receipt encoding with optional `reason_code`.
- Capture a stable `conversation_session_id` once per admitted turn as
  `gateway_session_key or agent.session_id`. Use the `X-Hermes-Session-Key`
  App scope for Local Server traffic and pass the same captured value to both
  pre- and transform-hooks even when compression rotates `agent.session_id`.
- Key proposal, dismiss/dedup, durable preference, and pending-receipt state by
  that stable conversation scope; keep `agent.session_id` as transcript
  lineage only when a gateway key exists.
- Read the exact request metadata capability
  `creation_action_receipt_transport: canonical_final_v1` as per-turn,
  server-only hook context; do not forward it to the model/provider.
- On `api_server`, keep creation-governor suppressed for missing or unknown
  capability values. Emit `action_receipts: true` only on a capable
  recommendation turn.
- Require the same exact capability again on an action turn for a
  receipt-capable card and fail before any side effect after a downgrade.
- Register `/v1/chat/completions/canonical-final-v1` as an exclusive admission
  endpoint for receipt-capable actions. Validate both the structured action and
  the exact metadata before Agent construction or model execution; ordinary
  and legacy traffic remains on `/v1/chat/completions`.
- Record pending results by `turn_id` in creation-governor state.
- Return structured internal action handling outcomes instead of inferring
  acceptance from whether prompt context is non-empty.
- Preserve PR #244's native Agent/Skill/Task routing.
- Pass the existing current turn id into `transform_llm_output` hooks.
- Strip model-forged result markers and append at most one trusted marker.
- Carry the canonical transformed final response on the streaming terminal
  chunk whenever the governor emits or sanitizes a marker, including when the
  canonical bytes equal the model draft.
- Close create proposals when the validated native-route handoff is accepted;
  do not reopen them from a generic failed, interrupted, incomplete, or empty
  terminal.
- Reject malformed structured wrappers without falling through to legacy text
  matching. Require a non-empty proposal id for every action and allow
  conversation-level unmute only while that scoped session is muted.
- Reconcile a retry after a committed mute or unmute as accepted when the same
  scoped session already has the explicit durable target value; do not rewrite
  SQLite and do not apply this rule to create.
- Model preference reads as four states: `muted`, `unmuted`, `absent`, and
  `read_error`. Only the exact explicit target reconciles; `absent` continues
  normal proposal validation, while `read_error` performs no mutation and
  returns retryable `preference_not_persisted` without consuming the proposal.

### Zettlab Web

- Parse the optional receipt `reason_code`.
- Let an exact valid receipt outrank generic assistant terminal status.
- Keep exact-turn retry isolation and bounded no-turn legacy recovery.
- Project terminal `proposal_not_actionable` as unavailable without a retry.
- Keep preference persistence failures retryable only while actionable.
- Project a terminal create with no trustworthy receipt as uncertain and
  non-retryable, rather than claiming failure and risking duplicate creation.
- Use action-specific failure copy rather than describing every rejection as a
  transport failure.
- Preserve settled proposal and accepted session preference state across later
  failed attempts, expiry, refresh, and pagination.

### Zettlab Local Server

- Send `creation_action_receipt_transport: canonical_final_v1` as request
  metadata on every recommendation and action turn only after canonical
  replacement is deployed.
- Preserve the stable App session in `X-Hermes-Session-Key`; do not replace it
  with a rotated Hermes transcript-lineage session id.
- Detect a validated receipt-capable action only for transport admission and
  send it exclusively to
  `/v1/chat/completions/canonical-final-v1`. Keep ordinary and legacy turns on
  `/v1/chat/completions`.
- Never fall back to the legacy endpoint after a versioned-endpoint 404, other
  4xx, timeout, connection loss, or uncertain response.
- Read Hermes' terminal `hermes.canonical_final_response` extension.
- Replace the accumulated streamed draft before emitting `turn.end.final_text`.
- Remove hidden action-result markers from a terminal draft when an older or
  disconnected Hermes did not provide a canonical final response.
- Persist and paginate the canonical text, outer `turn_id`, and stable message
  ids. Do not interpret creation domain outcomes beyond the minimal action
  wrapper check required to select the protected transport.

## Test-first sequence

1. Hermes codec and behavior tests fail for capability declaration, accepted
   and rejected receipts, reason codes, exact-turn ownership, forged markers,
   stable conversation ownership across session-id rotation, four-state
   preference reads, and pending-result cleanup.
2. Hermes finalizer integration fails until `turn_id` reaches the transform
   hook and transformed suffixes remain durable.
3. Hermes API integration fails until the versioned endpoint rejects
   unsupported admission before Agent/model execution and accepts only exact
   capable actions.
4. Local transport integration fails until capable actions use only the
   versioned endpoint and all failures prove no legacy fallback.
5. Web parser and state tests fail for reason codes, receipt precedence,
   terminal rejection without retry, and settled-state monotonicity.
6. Web component tests fail for the corresponding visible text and controls.
7. Implement the smallest production changes that make the above tests pass.

## Required Hermes matrix

- Agent, Skill, and Task create normal turns: accepted, proposal closed. A
  clarification before any tool call still counts because acceptance proves
  type-specific native routing, not tool execution or resource creation.
- Create empty, failed, interrupted, or incomplete after a validated handoff:
  accepted remains authoritative and the proposal stays closed.
- Expired, stale, replayed, wrong-owner, or mismatched action: rejected with
  `proposal_not_actionable`, current valid proposal unchanged.
- Dismiss latch succeeds: accepted even when narration later fails.
- Mute/unmute persistence succeeds: accepted even when narration later fails.
- A committed mute/unmute whose receipt was lost is accepted on a same-session
  retry without another write, including after restart; an absent or opposite
  preference row is not a reconciliation match.
- Preference lookup returns explicit `muted`, explicit `unmuted`, `absent`, or
  `read_error`. Read failure is not coerced to unmuted/absent: it is rejected
  with `preference_not_persisted`, performs no write, and leaves the current
  proposal retryable. Recommendation display also fails closed on read error.
- Mute/unmute write fails: rejected with `preference_not_persisted`, prior
  preference unchanged.
- Same proposal attempted in two turns: each transform consumes only its own
  pending result.
- Model-authored result marker: stripped and replaced by one governor marker.
- A byte-identical model marker and governor receipt still produces terminal
  canonical delivery; an ordinary reply does not.
- Missing, non-string, or unknown API transport capability suppresses new
  recommendations. An exact capability advertises `action_receipts: true`.
- A capable card followed by an action request with missing or unknown
  capability fails before proposal, preference, or native creation mutation.
- Concurrent and sequential requests do not inherit another turn's transport
  capability; pre- and transform-hooks see the same current-turn value.
- Out-of-place compression may rotate `agent.session_id` between hooks without
  changing proposal ownership, durable preference scope, or pending-receipt
  lookup. Parent and child transcript lineage ids share the stable
  `X-Hermes-Session-Key` App scope.
- The versioned endpoint plus exact metadata and a valid action reaches the
  governor. Missing, non-string, unknown, or omitted capability, malformed or
  non-action content, and the same request against old Hermes all fail before
  Agent/model invocation and before side effects.
- Malformed structured wrapper: no legacy text fallback and no action mutation.
- Unmute while muted: durably committed. Unmute while not muted is accepted
  only when an explicit durable unmuted preference proves retry
  reconciliation; otherwise it is rejected.

## Deployment fence

Use both the versioned endpoint and exact metadata as the executable fence.
Metadata alone is insufficient because old Hermes can ignore it on the legacy
endpoint. Roll out in this order:

1. Web can render legacy and capable cards and can disable a capable action
   when the current Local transport is below the canonical-final minimum.
2. Local Server ships canonical replacement, marker stripping, protected-route
   selection, and no fallback, with receipt opt-in still disabled.
3. Hermes ships stable conversation scope, four-state preference reads, and
   `/v1/chat/completions/canonical-final-v1` admission.
4. Enable Local request metadata and capable-action routing.

| Local Server | Hermes | Expected oracle |
| --- | --- | --- |
| old | old | Legacy traffic only |
| old | new | Legacy endpoint works; missing capability suppresses the new governor |
| new | old | Ordinary traffic works; capable action receives 404 before model and is never retried on the legacy endpoint |
| new | new | Exact capable action uses the versioned endpoint and receives canonical final v1 |

Hermes rollback is safe because the protected route disappears and yields 404
before model execution. Local rollback requires an operational minimum-version
fence while capable cards remain actionable, or a current-Local capability
check in Web that disables the click. An old Local must never transport a
receipt-capable wrapper on the legacy endpoint. Web/history provenance is not a
substitute for either transport fence. Unsafe pre-fence history needs separate
quarantine or migration.

## Required Web matrix

- Exact-turn accepted and rejected receipts.
- Wrong turn, proposal, or action ignored.
- A later retry does not truncate the earlier attempt scan.
- Accepted receipt remains authoritative beside generic error/abort metadata.
- Missing create receipt plus authoritative terminal is uncertain and has no
  retry control; incomplete newer history remains unconfirmed.
- Terminal `proposal_not_actionable` hides retry.
- `preference_not_persisted` allows retry only before expiry.
- Accepted create remains settled after a later rejected duplicate.
- Rejected mute does not cover a settled card.
- Rejected unmute does not cancel the last accepted mute.
- Legacy capability and no-turn history remain bounded and compatible.
- Refresh, expiry, and cross-page history preserve the live result.

## Cross-repository acceptance journeys

1. Valid create: click a card, observe the native flow, an accepted same-turn
   receipt, and Web copy that says handed to Hermes rather than created.
2. Expired or replayed create: observe terminal rejection with no retry and no
   mutation of a newer valid card.
3. Preference persistence failure: inject the database failure, observe a
   rejected same-turn receipt, unchanged preference, and retryable Web UI.
4. Interrupted create: observe the accepted handoff or an uncertain terminal,
   with no card retry that could duplicate a committed downstream resource.
5. Refresh and pagination: reload after each result and load history across a
   page boundary; observe identical card state with stable message identity.
6. Compression rotation: rotate Hermes transcript lineage between pre- and
   transform-hooks; observe one exact receipt and unchanged App-scope proposal
   and preference ownership.
7. Preference read fault: inject SQLite locked/unavailable/corrupt read;
   observe `preference_not_persisted`, no write or proposal consumption, then a
   successful retry after recovery. Contrast with an absent row, which does not
   prove prior unmute.
8. Hermes rollback: display a capable card through new Local/new Hermes, roll
   Hermes back, click the card, and observe route-level 404, zero model/governor
   calls, and no legacy retry. For Local rollback, observe the card disabled by
   the minimum-version/capability fence before any send.

## Gates before publication

- Hermes focused plugin tests and finalizer integration tests.
- Hermes relevant full test command and formatting/static checks.
- Web focused parser, resolver, converter, and component tests.
- Web typecheck, lint/static check, and repository gate required by PR #382.
- Local Server transport/history regression proof or an explicit source-backed
  zero-change gate, including route selection and no-fallback integration.
- Real HTTP proof that old Hermes returns 404 for the protected route before
  Agent/model construction and that new Hermes rejects wrong metadata before
  side effects.
- Manual cross-repository journey evidence for the eight scenarios above.
- Adversarial review bound to the exact final SHAs.

Only after all gates pass: commit with clear Simplified Chinese descriptions,
push both branches, open/update the Hermes PR and Web PR #382, and trigger the
review bot. Do not merge until current-SHA remote checks and review findings are
clear.
