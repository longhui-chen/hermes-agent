"""Shared client for Zettlab media generation via local-server ai-proxy.

Hermes runs on the device and must not hold a cloud user token. The local-server
loopback ai-proxy mints the device IoT credential and forwards these calls to
zettlab-ai-gateway.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:9090/api/v1/ai-proxy/v1"
CAPABILITY_TIMEOUT = 5.0
REQUEST_TIMEOUT = 30.0
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
    configured = os.environ.get("ZETTLAB_AI_PROXY_BASE_URL") or _config_section(media_type).get("base_url")
    raw = str(configured or DEFAULT_BASE_URL).strip().rstrip("/")
    return raw or DEFAULT_BASE_URL


def get_capabilities(media_type: Optional[str] = None) -> Dict[str, Any]:
    mt = media_type or "image"
    resp = requests.get(
        f"{base_url(mt)}/media/generation-capabilities",
        timeout=CAPABILITY_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ZettlabMediaError("media capability response is not a JSON object")
    return data


def action_headers() -> Dict[str, str]:
    token = os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "").strip()
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
    configured = _config_section(media_type).get("model")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    models = list_models(media_type)
    if models:
        return str(models[0]["id"])
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
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
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
    resp = requests.post(
        f"{base_url(media_type)}/media/generation-jobs",
        json=body,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
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
    while time.monotonic() < deadline:
        time.sleep(max(0.2, poll_interval))
        resp = requests.get(
            f"{base_url(media_type)}/media/generation-jobs/{job_id}",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        job = resp.json()
        if not isinstance(job, dict):
            raise ZettlabMediaError("media generation poll response is not a JSON object")
        status = str(job.get("status") or "")
        if status == "done":
            return job
        if status in {"failed", "cancelled"}:
            msg = str(job.get("error_message") or job.get("error_code") or status)
            raise ZettlabMediaError(f"media generation job {status}: {msg}")
    raise ZettlabMediaError("media generation timed out")


def first_asset_url(job: Dict[str, Any]) -> str:
    assets = job.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ZettlabMediaError("media generation completed without assets")
    first = assets[0]
    if not isinstance(first, dict):
        raise ZettlabMediaError("media generation asset has invalid shape")
    url = first.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    shortcut = job.get("image") or job.get("video")
    if isinstance(shortcut, str) and shortcut.strip():
        return shortcut.strip()
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
