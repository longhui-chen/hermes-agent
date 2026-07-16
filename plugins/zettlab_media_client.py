"""Shared client for Zettlab media generation via local-server ai-proxy.

Hermes runs on the device and must not hold a cloud user token. The local-server
loopback ai-proxy mints the device IoT credential and forwards these calls to
zettlab-ai-gateway.
"""

from __future__ import annotations

import ipaddress
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
from agent.secret_scope import get_secret
from tools.interrupt import is_interrupted

_SESSION = requests.Session()
_SESSION.trust_env = False

DEFAULT_BASE_URL = "http://127.0.0.1:9090/api/v1/ai-proxy/v1"
CAPABILITY_TIMEOUT = 5.0
REQUEST_TIMEOUT = 30.0
POLL_RETRY_LIMIT = 3
ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"


class ZettlabMediaError(RuntimeError):
    """Raised when the local media generation proxy cannot satisfy a request."""


def _config_section(media_type: str) -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get(f"{media_type}_gen") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            return {}
        zettlab = section.get("zettlab")
        if isinstance(zettlab, dict):
            merged = dict(section)
            merged.update(zettlab)
            return merged
        return section
    except Exception:
        return {}


def base_url(media_type: str) -> str:
    explicit = str(get_secret("ZETTLAB_AI_PROXY_BASE_URL", "") or "").strip()
    configured = explicit or _config_section(media_type).get("base_url")
    if configured:
        raw = str(configured).strip().rstrip("/")
    else:
        append_url = str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
        if append_url:
            parsed_append = urlparse(append_url)
            raw = urlunparse((
                parsed_append.scheme,
                parsed_append.netloc,
                "/api/v1/ai-proxy/v1",
                "",
                "",
                "",
            ))
        else:
            raw = DEFAULT_BASE_URL
    raw = raw or DEFAULT_BASE_URL
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (
        parsed.scheme not in {"http", "https"}
        or not loopback
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port == -1
    ):
        raise ZettlabMediaError("Zettlab ai-proxy base_url must be a plain loopback HTTP(S) URL")
    return raw


def get_capabilities(media_type: Optional[str] = None) -> Dict[str, Any]:
    mt = media_type or "image"
    resp = _SESSION.get(
        f"{base_url(mt)}/media/generation-capabilities",
        timeout=CAPABILITY_TIMEOUT,
        allow_redirects=False,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ZettlabMediaError("media capability response is not a JSON object")
    return data


def action_headers() -> Dict[str, str]:
    token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    if not token:
        raise ZettlabMediaError("ZETTLAB_AGENT_ACTION_TOKEN is required for media generation")
    return {ACTION_TOKEN_HEADER: token}


def type_capability(media_type: str) -> Dict[str, Any]:
    caps = get_capabilities(media_type)
    section = caps.get(media_type)
    return section if isinstance(section, dict) else {}


def list_models(media_type: str) -> List[Dict[str, Any]]:
    section = type_capability(media_type)
    if section.get("enabled") is False:
        return []
    models = section.get("models")
    if not isinstance(models, list):
        return []
    return [m for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)]


def default_model(media_type: str) -> Optional[str]:
    section = type_capability(media_type)
    return _resolve_model_from_section(media_type, section)


def resolve_model(media_type: str, requested: Optional[str] = None) -> Optional[str]:
    model_id, _ = resolve_model_with_capability(media_type, requested)
    return model_id


def resolve_model_with_capability(
    media_type: str,
    requested: Optional[str] = None,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    section = type_capability(media_type)
    model_id = _resolve_model_from_section(media_type, section, requested)
    models = section.get("models")
    if model_id and isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
                return model_id, model
    return model_id, None


def selected_model_capability(media_type: str) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    section = type_capability(media_type)
    model_id = _resolve_model_from_section(media_type, section)
    models = section.get("models")
    if model_id and isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
                return section, model
    return section, None


def _resolve_model_from_section(
    media_type: str,
    section: Dict[str, Any],
    requested: Optional[str] = None,
) -> Optional[str]:
    if section.get("enabled") is False:
        return None
    gateway_default = section.get("default_model")
    models = section.get("models")
    if isinstance(models, list):
        model_ids = [
            model.get("id").strip()
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("id"), str)
            and model.get("id").strip()
        ]
        explicit = str(requested or "").strip()
        if explicit:
            return explicit if explicit in model_ids else None
        configured = _config_section(media_type).get("model")
        if isinstance(configured, str) and configured.strip():
            configured_id = configured.strip()
            return configured_id if configured_id in model_ids else None
        if "default_model" in section:
            if isinstance(gateway_default, str) and gateway_default.strip() in model_ids:
                return gateway_default.strip()
            return None
        # Compatibility with gateways that predate the explicit default_model
        # field. Reuse this same capability snapshot so a transient second GET
        # cannot invalidate an otherwise usable response.
        if model_ids:
            return model_ids[0]
    return None


def is_available(media_type: str) -> bool:
    try:
        section = type_capability(media_type)
    except Exception:
        return False
    return bool(section.get("enabled")) and bool(section.get("models"))


def validate_remote_url(value: Optional[str], *, label: str) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    try:
        is_ip_literal = bool(host) and ipaddress.ip_address(host) is not None
    except ValueError:
        is_ip_literal = False
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
        or port not in {None, 443}
        or not host
        or host == "localhost"
        or host.endswith(".localhost")
        or is_ip_literal
    ):
        raise ZettlabMediaError(f"{label} must be an https URL for Zettlab media generation")
    return raw


def remote_inputs(
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    primary = validate_remote_url(image_url, label="image_url")
    if primary:
        out.append({"url": primary, "role": "source"})
    for idx, ref in enumerate(reference_image_urls or []):
        normalized = validate_remote_url(ref, label=f"reference_image_urls[{idx}]")
        if normalized:
            out.append({"url": normalized, "role": "reference"})
    return out


def create_and_wait(
    *,
    media_type: str,
    model: str,
    prompt: str,
    payload: Dict[str, Any],
    timeout_seconds: Optional[int] = None,
    poll_interval: float = 2.0,
) -> Dict[str, Any]:
    body = {
        "media_type": media_type,
        "model": model,
        "prompt": prompt,
        **payload,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Scene-Type": "media_generation",
        "X-Step-Title": "media_generation",
        **action_headers(),
    }
    resp = _SESSION.post(
        f"{base_url(media_type)}/media/generation-jobs",
        json=body,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )
    resp.raise_for_status()
    job = resp.json()
    if not isinstance(job, dict):
        raise ZettlabMediaError("media generation job response is not a JSON object")
    if job.get("status") == "done":
        return job

    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ZettlabMediaError("media generation job response did not include job_id")

    deadline = time.monotonic() + float(timeout_seconds or _timeout_from_capability(media_type))
    poll_failures = 0
    try:
        while time.monotonic() < deadline:
            _interruptible_sleep(max(0.2, poll_interval))
            try:
                resp = _SESSION.get(
                    f"{base_url(media_type)}/media/generation-jobs/{job_id}",
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=False,
                )
                resp.raise_for_status()
                job = resp.json()
            except requests.RequestException:
                poll_failures += 1
                if poll_failures > POLL_RETRY_LIMIT:
                    raise
                continue
            poll_failures = 0
            if not isinstance(job, dict):
                raise ZettlabMediaError("media generation poll response is not a JSON object")
            status = str(job.get("status") or "")
            if status == "done":
                return job
            if status in {"failed", "cancelled"}:
                msg = str(job.get("error_message") or job.get("error_code") or status)
                raise ZettlabMediaError(f"media generation job {status}: {msg}")
        raise ZettlabMediaError("media generation timed out")
    except BaseException:
        _delete_job(media_type, job_id, headers)
        raise


def _interruptible_sleep(delay: float) -> None:
    deadline = time.monotonic() + delay
    while time.monotonic() < deadline:
        if is_interrupted():
            raise ZettlabMediaError("media generation interrupted")
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def _delete_job(media_type: str, job_id: str, headers: Dict[str, str]) -> None:
    try:
        _SESSION.delete(
            f"{base_url(media_type)}/media/generation-jobs/{job_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        ).raise_for_status()
    except Exception:
        pass


def first_asset_url(job: Dict[str, Any]) -> str:
    assets = job.get("assets")
    if isinstance(assets, list) and assets and isinstance(assets[0], dict):
        url = assets[0].get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    shortcut = job.get("image") or job.get("video")
    if isinstance(shortcut, str) and shortcut.strip():
        return shortcut.strip()
    if not isinstance(assets, list) or not assets:
        raise ZettlabMediaError("media generation completed without assets")
    if not isinstance(assets[0], dict):
        raise ZettlabMediaError("media generation asset has invalid shape")
    raise ZettlabMediaError("media generation asset has no retrievable URL")


def _timeout_from_capability(media_type: str) -> int:
    try:
        limits = type_capability(media_type).get("limits")
        if isinstance(limits, dict):
            provider_timeout = int(limits.get("provider_timeout_seconds") or 0)
            finalization_timeout = int(limits.get("finalization_timeout_seconds") or 0)
            total = provider_timeout + finalization_timeout
            if total > 0:
                return total
    except Exception:
        pass
    return 30 * 60 if media_type == "video" else 10 * 60
