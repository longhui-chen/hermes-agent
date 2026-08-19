"""Relay media client — gateway↔connector media plane (Phase 2). EXPERIMENTAL.

The relay wire contract carries media BY REFERENCE, never by value: an inbound
event's ``media_urls`` name connector re-hosted attachments
(``{connector}/relay/media/{id}``), and an outbound ``send_media`` op names a
``source_url`` the connector resolves back to bytes. This module is the
gateway-side HTTP client for that plane:

  - ``download(url)``  → GET a re-hosted attachment to a local temp file (the
    agent's vision/file tools consume LOCAL paths, matching every native
    adapter's inbound media behaviour).
  - ``upload(path)``   → POST local file bytes to ``/relay/media``; returns the
    ``/relay/media/{id}`` reference for a subsequent ``send_media`` op. This is
    how a locally-generated artifact (image_generate output, TTS voice note,
    a document) crosses to the connector WITHOUT the gateway needing a public
    URL.

Both requests present the SAME per-gateway signed bearer the WS upgrade uses
(``make_upgrade_token``, gateway/relay/auth.py — the channel authenticator; the
connector authenticates it with ``authenticateGatewayBearer`` on the mirrored
routes). Uploads are
per-gateway-owned on the connector (only this gateway can reference the id
back); downloads accept both this gateway's uploads and connector ingest
re-hosts.

Transport is stdlib ``urllib`` run in a thread executor (the same dependency
posture as ``_post_provision`` — the relay lane adds no HTTP client deps).
EXPERIMENTAL: may change without a deprecation cycle (docs/relay-connector-contract.md).
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from gateway.relay.auth import make_upgrade_token

logger = logging.getLogger(__name__)

# Mirror the connector's MEDIA_MAX_BYTES (mediaStore.ts) so an oversized local
# artifact fails fast here instead of round-tripping to a connector 413.
MEDIA_MAX_BYTES = 25 * 1024 * 1024

_REQUEST_TIMEOUT_S = 30.0


def media_base_url(relay_dial_url: str) -> str:
    """Map the ``ws(s)://…/relay`` dial URL to the ``http(s)://…`` base.

    Same host derivation as ``_provision_url`` (gateway/relay/__init__.py):
    scheme ws→http / wss→https, trailing ``/relay`` stripped.
    """
    raw = (relay_dial_url or "").strip().rstrip("/")
    if raw.startswith("ws://"):
        raw = "http://" + raw[len("ws://") :]
    elif raw.startswith("wss://"):
        raw = "https://" + raw[len("wss://") :]
    if raw.endswith("/relay"):
        raw = raw[: -len("/relay")]
    return raw


class RelayMediaClient:
    """Authenticated client for the connector's ``/relay/media`` routes."""

    def __init__(
        self,
        base_url: str,
        gateway_id: Optional[str],
        secret: Optional[str],
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._base_origin = self._origin(self._base_url)
        self._gateway_id = gateway_id or ""
        self._secret = secret or ""

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int] | None:
        try:
            parsed = urllib.parse.urlsplit(url)
            scheme = parsed.scheme.lower()
            if scheme not in {"http", "https"} or not parsed.hostname:
                return None
            port = parsed.port or (443 if scheme == "https" else 80)
            return scheme, parsed.hostname.lower(), port
        except (TypeError, ValueError):
            return None

    @property
    def enabled(self) -> bool:
        """True when the client can authenticate (per-gateway creds present)."""
        return bool(self._base_url and self._gateway_id and self._secret)

    def _bearer(self) -> str:
        return make_upgrade_token(self._gateway_id, self._secret)

    def is_relay_media_url(self, url: str) -> bool:
        """Is ``url`` a connector re-host reference (needs our bearer to GET)?"""
        try:
            parsed = urllib.parse.urlsplit(url or "")
        except (TypeError, ValueError):
            return False
        return (
            self._base_origin is not None
            and self._origin(url) == self._base_origin
            and parsed.path.startswith("/relay/media/")
        )

    async def upload(
        self,
        file_path: str,
        *,
        mime: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """POST local file bytes to ``/relay/media``; return the reference URL.

        Returns the ``{base}/relay/media/{id}`` reference for a ``send_media``
        op's ``source_url``, or None on any failure (callers fall back to their
        pre-media behaviour — media delivery is best-effort by design).
        """
        if not self.enabled:
            return None
        path = Path(file_path)
        try:
            with path.open("rb") as handle:
                data = handle.read(MEDIA_MAX_BYTES + 1)
        except OSError:
            logger.warning("relay media upload: cannot read %s", file_path)
            return None
        if not data or len(data) > MEDIA_MAX_BYTES:
            logger.warning(
                "relay media upload: %s size %d outside (0, %d]",
                file_path,
                len(data),
                MEDIA_MAX_BYTES,
            )
            return None
        content_type = (
            mime
            or mimetypes.guess_type(filename or path.name)[0]
            or "application/octet-stream"
        )
        headers = {
            "Authorization": f"Bearer {self._bearer()}",
            "Content-Type": content_type,
            "X-Media-Filename": re.sub(
                r"[\x00-\x1f\x7f]", "", filename or path.name
            )[:255],
        }
        url = f"{self._base_url}/relay/media"

        def _post() -> Optional[str]:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                    import json

                    raw = resp.read(65537)
                    if len(raw) > 65536:
                        raise ValueError("relay media upload response too large")
                    body = json.loads(raw.decode("utf-8"))
                    media_id = body.get("id")
                    if not media_id:
                        return None
                    return f"{self._base_url}/relay/media/{media_id}"
            except (urllib.error.URLError, ValueError, OSError) as exc:
                from gateway.platforms.base import safe_exc

                logger.warning("relay media upload failed: %s", safe_exc(exc))
                return None

        return await asyncio.get_running_loop().run_in_executor(None, _post)

    async def download(self, url: str, *, suggested_name: Optional[str] = None) -> Optional[str]:
        """GET a re-hosted attachment to a local temp file; return its path.

        Presents the per-gateway bearer for connector re-host URLs; plain
        public URLs (e.g. a Discord CDN pass-through) are fetched without it.
        Returns None on any failure (the event then keeps the remote URL, and
        downstream consumers that need a local file skip it — best-effort).
        """
        if not url:
            return None
        needs_auth = self.is_relay_media_url(url)
        if needs_auth and not self.enabled:
            return None
        headers = {}
        if needs_auth:
            headers["Authorization"] = f"Bearer {self._bearer()}"

        from gateway.platforms.base import (
            _read_httpx_body_with_limit,
            _ssrf_redirect_guard,
            safe_exc,
            safe_url_for_log,
        )
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url

        if not needs_auth and not is_safe_url(url):
            logger.warning(
                "relay media download blocked unsafe URL: %s", safe_url_for_log(url)
            )
            return None

        try:
            if needs_auth:
                import httpx

                client_cm = httpx.AsyncClient(
                    timeout=_REQUEST_TIMEOUT_S,
                    follow_redirects=False,
                )
            else:
                client_cm = create_ssrf_safe_async_client(
                    timeout=_REQUEST_TIMEOUT_S,
                    follow_redirects=True,
                    event_hooks={"response": [_ssrf_redirect_guard]},
                )
            async with client_cm as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    if needs_auth and getattr(resp, "is_redirect", False):
                        raise ValueError("authenticated relay media redirect refused")
                    resp.raise_for_status()
                    data = await _read_httpx_body_with_limit(
                        resp,
                        media_type="relay inbound media",
                        max_bytes=MEDIA_MAX_BYTES,
                    )
                    if not data:
                        return None
                    name = suggested_name or ""
                    if not name:
                        cd = resp.headers.get("content-disposition") or ""
                        if "filename=" in cd:
                            name = cd.split("filename=", 1)[1].strip().strip('"')
                    ext = Path(name).suffix if name else ""
                    if not ext:
                        mime = (resp.headers.get("content-type") or "").split(";")[0]
                        ext = mimetypes.guess_extension(mime) or ".bin"
                    fd, tmp_path = tempfile.mkstemp(
                        prefix="relay_media_", suffix=ext,
                    )
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    return tmp_path
        except Exception as exc:
            logger.warning(
                "relay media download failed for %s: %s",
                safe_url_for_log(url),
                safe_exc(exc),
            )
            return None


__all__ = ["RelayMediaClient", "media_base_url", "MEDIA_MAX_BYTES"]
