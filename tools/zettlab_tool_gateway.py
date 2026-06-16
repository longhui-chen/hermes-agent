"""Zettlab-hosted managed-tool gateway helpers.

This is the Zettlab equivalent of :mod:`tools.managed_tool_gateway`. Both let a
vendor tool (Browser Use, …) run without the agent holding the commercial key:
a gateway fronts the vendor, injects the real key, and bills/audits by caller.
The difference is who operates the gateway and how the caller is authorized.

- Nous path (managed_tool_gateway): gated by a Nous Portal subscription /
  tool-pool entitlement; the agent carries a Nous OAuth token.
- Zettlab path (this module): used on Zettlab devices, where Hermes runs as a
  child of local-server. local-server injects an explicit per-vendor gateway
  URL (e.g. ``BROWSER_USE_GATEWAY_URL`` → its loopback proxy) plus the shared
  ``ZETTLAB_TOOL_GATEWAY_TOKEN`` placeholder. Authorization is the device's IoT
  identity, attached by local-server downstream — NOT a Nous account — so this
  path deliberately does NOT consult ``managed_nous_tools_enabled()``. Presence
  of an explicit gateway URL is itself the signal that a managed gateway is
  configured.

This is resolved once, centrally, from :func:`resolve_managed_tool_gateway`
(it returns the Zettlab config before the Nous entitlement gate), so every
gateway-backed tool gets the Zettlab path with no per-provider wiring.

A dedicated ``ZETTLAB_TOOL_GATEWAY_TOKEN`` (not Nous' ``TOOL_GATEWAY_USER_TOKEN``)
is used so the placeholder can never leak into Nous token-read paths.

Resolution is purely from explicit env (no nousresearch.com default): a device
without the gateway configured gets ``None`` and the caller falls back to the
Nous path or local execution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZettlabToolGatewayConfig:
    vendor: str
    gateway_origin: str
    # token is a local placeholder forwarded as the vendor auth header; the
    # gateway authenticates the device by its IoT identity, so this is never a
    # cloud secret. Required only because the vendor providers expect a
    # non-empty value.
    token: str


def explicit_vendor_gateway_url(vendor: str) -> str:
    """Return the explicitly-configured ``{VENDOR}_GATEWAY_URL``, or "".

    Unlike :func:`managed_tool_gateway.build_vendor_gateway_url`, this never
    falls back to a shared domain or the nousresearch.com default: only an
    explicit per-vendor override counts as a Zettlab-managed gateway.
    """
    vendor_key = f"{vendor.upper().replace('-', '_')}_GATEWAY_URL"
    return os.getenv(vendor_key, "").strip().rstrip("/")


def resolve_zettlab_tool_gateway(vendor: str) -> Optional[ZettlabToolGatewayConfig]:
    """Resolve the Zettlab-managed gateway for a vendor from explicit env.

    Returns ``None`` unless both an explicit ``{VENDOR}_GATEWAY_URL`` and the
    ``ZETTLAB_TOOL_GATEWAY_TOKEN`` placeholder are present (both injected by
    local-server when ``browser_use.enabled``). No Nous entitlement check.
    """
    origin = explicit_vendor_gateway_url(vendor)
    if not origin:
        return None
    token = os.getenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "").strip()
    if not token:
        return None
    return ZettlabToolGatewayConfig(vendor=vendor, gateway_origin=origin, token=token)
