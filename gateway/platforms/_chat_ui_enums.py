"""Generated Chat UI contract enums (B1).

Source: zettlab-product-dev contract snapshot. Do not edit manually; regenerate
when the pinned snapshot changes.
"""

TERMINAL_STATES = frozenset({"expired", "cancelled", "runtime_lost"})
STATE_REASONS = frozenset({
    "timeout", "turn_interrupted", "session_reset", "delivery_failed",
    "request_unknown", "gateway_restart",
})
SENTINEL_REASONS = frozenset(set(STATE_REASONS) | {"caller_inactive"})
PROVIDER_ERROR_CODE_MAP = {"insufficient_quota": "user_credits_insufficient"}
