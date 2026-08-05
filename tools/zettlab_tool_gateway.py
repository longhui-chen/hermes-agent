"""Zettlab-hosted managed-tool gateway helpers.

This is the Zettlab equivalent of :mod:`tools.managed_tool_gateway`. Both let a
vendor tool (Browser Use, …) run without the agent holding the commercial key:
a gateway fronts the vendor, injects the real key, and bills/audits by caller.
The difference is who operates the gateway and how the caller is authorized.

- Nous path (managed_tool_gateway): gated by a Nous Portal subscription /
  tool-pool entitlement; the agent carries a Nous OAuth token.
- Zettlab path (this module): used on Zettlab devices, where Hermes runs as a
  child of local-server. local-server injects the generic callback URL
  (``ZET_CHAT_APPEND_URL``) and the per-agent action token
  (``ZETTLAB_AGENT_ACTION_TOKEN``). Hermes derives the loopback gateway URL for
  each supported vendor from that local-server origin. Authorization is the
  action token accepted by local-server — NOT a Nous account — so this path
  deliberately does NOT consult ``managed_nous_tools_enabled()``.

This is resolved once, centrally, from :func:`resolve_managed_tool_gateway`
(it returns the Zettlab config before the Nous entitlement gate), so every
gateway-backed tool gets the Zettlab path with no per-provider wiring.

Resolution is purely from local-server env (no nousresearch.com default): a
device without local-server callback/action env gets ``None`` and the caller
falls back to the Nous path or local execution.
"""

from __future__ import annotations

import logging
import ipaddress
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

_LOCAL_SERVER_ANCHOR_ENVS = (
    "ZET_CHAT_APPEND_URL",
    "ZETTLAB_AGENT_SHARE_ACTION_URL",
)
_ACTION_TOKEN_ENV = "ZETTLAB_AGENT_ACTION_TOKEN"
_VENDOR_GATEWAY_PATHS = {
    "browser-use": "/api/v1/browser-use",
    # OpenAI SDK appends /v1/audio/speech to this origin.
    "openai-tts": "/api/v1/ai-proxy",
}


@dataclass(frozen=True)
class ZettlabToolGatewayConfig:
    vendor: str
    gateway_origin: str
    # token is a local placeholder forwarded as the vendor auth header; the
    # gateway authenticates the device by its IoT identity, so this is never a
    # cloud secret. Required only because the vendor providers expect a
    # non-empty value.
    token: str


def _local_server_origin() -> str:
    for env_key in _LOCAL_SERVER_ANCHOR_ENVS:
        raw = str(get_secret(env_key, "") or "").strip()
        if not raw:
            continue
        parts = None
        try:
            parts = urlsplit(raw)
            hostname = parts.hostname or ""
            # Accessing port validates malformed values such as :not-a-port.
            _ = parts.port
        except ValueError:
            hostname = ""
        if (
            parts is None
            or parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.username is not None
            or parts.password is not None
            or not _is_loopback_host(hostname)
        ):
            logger.debug("Ignoring malformed %s for Zettlab gateway: %r", env_key, raw)
            continue
        return urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")
    return ""


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def local_server_gateway_url(vendor: str) -> str:
    """Return the local-server loopback gateway URL for a supported vendor."""
    path = _VENDOR_GATEWAY_PATHS.get(vendor)
    if not path:
        return ""
    origin = _local_server_origin()
    if not origin:
        return ""
    return f"{origin}{path}"


def resolve_zettlab_tool_gateway(vendor: str) -> Optional[ZettlabToolGatewayConfig]:
    """Resolve the Zettlab-managed gateway for a vendor from local-server env.

    Returns ``None`` unless local-server has injected a valid callback/action
    environment and the vendor has a route in ``_VENDOR_GATEWAY_PATHS``. No Nous
    entitlement check.
    """
    origin = local_server_gateway_url(vendor)
    if not origin:
        return None
    token = str(get_secret(_ACTION_TOKEN_ENV, "") or "").strip()
    if not token:
        return None
    return ZettlabToolGatewayConfig(vendor=vendor, gateway_origin=origin, token=token)
