"""Hardened transport for loopback requests that carry device credentials.

Shared by call sites that send the per-agent action token to a local-server
loopback face (App Host internal face, NAS agent-search callback). Both faces
share the same server-side admission model (loopback-only + action token), so
one trust rule fits both. Three guarantees, kept in ONE place so they cannot
drift apart per call site:

- endpoint trust (:func:`is_trusted_loopback_http`): plain http + literal
  loopback host only. The faces are plain HTTP on loopback; any other scheme
  or host means the configured value was repointed somewhere the credential
  does not belong. A prefix test would accept ``127.attacker.example``.
- no environment proxies: HTTP_PROXY/ALL_PROXY (when NO_PROXY doesn't cover
  loopback) would forward the request — credential included — to whatever
  host the proxy names. Validating the URL is not enough; the transport
  itself must refuse the proxy (rationale shared with browser_camofox).
- no redirects: the loopback check constrains only the FIRST hop; following
  a 3xx would replay the credential against whatever the Location header
  names. The loopback faces never legitimately redirect, so any 3xx is an
  anomaly — urllib surfaces it as an HTTPError.
"""

import ipaddress
import urllib.request


def is_trusted_loopback_http(parts) -> bool:
    """Whether a urlsplit result is a plain-http, literal-loopback endpoint."""
    if parts.scheme != "http":
        return False
    host = (parts.hostname or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow ANY redirect (returning None makes urllib raise the
    original 3xx as an HTTPError instead of re-sending the request)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


HARDENED_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _RefuseRedirect()
)


def urlopen_hardened(req, timeout):
    """Open a credentialed loopback request: no env proxies, no redirects."""
    return HARDENED_OPENER.open(req, timeout=timeout)
