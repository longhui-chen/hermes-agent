# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install langfuse
hermes plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # then check Langfuse for a "Hermes turn" trace
```

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
HERMES_LANGFUSE_SN=WY210260528GT102  # device serial — see "Per-device fleets" below
```

## Per-device fleets

Running one Hermes gateway per device? Set `HERMES_LANGFUSE_SN` to the device
serial and every trace is tagged with the device it came from, three ways:

- **`user_id`** — each device shows up in Langfuse's **Users** view, so you get
  per-device cost / trace-count / latency dashboards for free. This scales to
  thousands of devices (high cardinality is fine here, unlike `environment`).
- **tag `sn:<serial>`** — one-click filtering in the traces list.
- **`metadata.device_sn`** — exact-match filtering and export.

On Zettlab devices `zettlab-local-server` injects this automatically from
`device.sn` when it spawns the per-agent gateway — you don't set it by hand.
Leave it unset and traces behave exactly as before (no `user_id`, no `sn:` tag).

> Don't overload `HERMES_LANGFUSE_ENV` with the serial for this — `environment`
> is a low-cardinality dimension (prod / staging / dev), not a per-device key.

## Disable

```bash
hermes plugins disable observability/langfuse
```
