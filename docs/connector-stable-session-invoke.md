# Connector stable session transport

The dedicated preset runner receives `ZETTLAB_CONNECTOR_SESSION_ID` from the task-local stable `HERMES_SESSION_KEY`. It never substitutes the compacted `HERMES_SESSION_ID`, another profile's process environment, or a default conversation. `ZETTLAB_AGENT_ACTION_TOKEN` authenticates the local profile; cloud IAM credentials remain outside subprocesses.

The preset sends `X-Zettlab-Connector-Session-Id` and `X-Zettlab-Agent-Action-Token` to the loopback broker. The broker no longer needs an active turn capability for this route. Cloud Session Gate still enforces the session switch, Agent policy and exact connection/account selection. The existing turn ID is forwarded for Action V2 correlation only; it is not looked up as an active grant.

Cron remains on its dedicated direct route. During the upgrade window the legacy capability is still exported for old presets. Deploy the matching local-server broker before the updated Hermes/presets packages; no device deployment is part of this source change.

## Overlay registry

Batch `connector-session`, upstream `none`: existing Zettlab-only private runner protocol. `tools/environments/local.py` adds profile authentication and stable selection fields to the existing allowlisted runner environment; `tools/terminal_tool.py` redacts the local profile token. No state machine or shared mutable state is added. Added kernel lines stay below the 60-line HR8 budget; each changed hunk has an overlay marker. Generic shell credential isolation remains in place.

Validation covers concurrent HTTP session routing, real restricted subprocess environment, Cron compatibility, and output redaction. Device/IAM/provider E2E remains pending, and local full QA/build is not run. Full cross-repository contract and HR1–HR8/HR-T1 assessment live in local-server `docs/superpowers/specs/2026-09-10-connector-stable-session-invoke.md`.
