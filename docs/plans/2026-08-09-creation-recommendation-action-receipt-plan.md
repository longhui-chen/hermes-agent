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
- Read the exact request metadata capability
  `creation_action_receipt_transport: canonical_final_v1` as per-turn,
  server-only hook context; do not forward it to the model/provider.
- On `api_server`, keep creation-governor suppressed for missing or unknown
  capability values. Emit `action_receipts: true` only on a capable
  recommendation turn.
- Require the same exact capability again on an action turn for a
  receipt-capable card and fail before any side effect after a downgrade.
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
- Read Hermes' terminal `hermes.canonical_final_response` extension.
- Replace the accumulated streamed draft before emitting `turn.end.final_text`.
- Remove hidden action-result markers from a terminal draft when an older or
  disconnected Hermes did not provide a canonical final response.
- Persist and paginate the canonical text, outer `turn_id`, and stable message
  ids without parsing the creation-recommendation payload itself.

## Test-first sequence

1. Hermes codec and behavior tests fail for capability declaration, accepted
   and rejected receipts, reason codes, exact-turn ownership, forged markers,
   and pending-result cleanup.
2. Hermes finalizer integration fails until `turn_id` reaches the transform
   hook and transformed suffixes remain durable.
3. Web parser and state tests fail for reason codes, receipt precedence,
   terminal rejection without retry, and settled-state monotonicity.
4. Web component tests fail for the corresponding visible text and controls.
5. Implement the smallest production changes that make the above tests pass.

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
  retry without another write, including after restart; an absent preference
  row is not an already-unmuted result.
- Mute/unmute persistence fails: rejected with
  `preference_not_persisted`, prior preference unchanged.
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
- Malformed structured wrapper: no legacy text fallback and no action mutation.
- Unmute while muted: durably committed. Unmute while not muted is accepted
  only when an explicit durable unmuted preference proves retry
  reconciliation; otherwise it is rejected.

## Deployment fence

Deploy and verify Local Server canonical replacement before it sends the exact
`creation_action_receipt_transport: canonical_final_v1` request metadata.
Hermes treats missing, malformed, and unknown values as absent, suppresses new
cards on `api_server`, and rechecks the capability before every receipt-capable
action side effect. This executable negotiation permits a mixed fleet without
trusting rollout intent. Web/history provenance is not a substitute because an
old Local ignores the canonical terminal extension before either consumer can
observe it. Any history already written through an unsafe pre-fence path needs
separate quarantine or migration.

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

## Gates before publication

- Hermes focused plugin tests and finalizer integration tests.
- Hermes relevant full test command and formatting/static checks.
- Web focused parser, resolver, converter, and component tests.
- Web typecheck, lint/static check, and repository gate required by PR #382.
- Local Server transport/history regression proof or an explicit source-backed
  zero-change gate.
- Manual cross-repository journey evidence for the five scenarios above.
- Adversarial review bound to the exact final SHAs.

Only after all gates pass: commit with clear Simplified Chinese descriptions,
push both branches, open/update the Hermes PR and Web PR #382, and trigger the
review bot. Do not merge until current-SHA remote checks and review findings are
clear.
