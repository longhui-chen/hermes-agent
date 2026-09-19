"""Cron-only Connector direct-execution route client.

Cron is a background caller, not a suspended Chat turn. It obtains a one-run
opaque route handle from its local-server parent; the handle never leaves the
device. The Server call is separately device-authenticated and re-checks owner,
Agent policy, provider connection, and reauth state for every MCP request.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_PREPARE_METHOD = "zettlab/cron/prepare-connector-execution"
_TIMEOUT_SECONDS = 10
_MAX_PRESET_MANIFESTS = 256
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ConnectorExecutionLeaseError(RuntimeError):
    """Compatibility name for the existing scheduler error boundary.

    The implementation no longer requests a Connector grant or lease token.
    """

    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.code = code or "task_connector_temporarily_unavailable"
        self.retryable = retryable


def enabled() -> bool:
    return _secret("ZETTLAB_CRON_CONNECTOR_DIRECT_ENABLED") == "1"


def connector_provider_for_skills(skills: Any) -> str:
    """Return the Connector provider a Cron job binds its direct route to.

    A Cron job may include ordinary skills alongside Connector skills. When
    the trusted preset manifests name exactly one Connector provider, the
    direct route stays bound to that provider (legacy behaviour). When they
    name none or several, ``""`` is returned and the caller prepares an
    Agent-level route instead: the local-server broker then resolves the
    provider per request (MCP tool-name prefix or the
    ``X-Zettlab-Connector-Provider`` header) and Server policy still decides
    every call. Nothing is rejected here any more.
    """
    if isinstance(skills, str):
        skills = [skills]
    skill_ids = {str(skill).strip().lower() for skill in (skills or []) if str(skill).strip()}
    if not skill_ids:
        return ""
    providers = {_preset_connector_providers().get(skill_id, "") for skill_id in skill_ids}
    providers.discard("")
    if len(providers) == 1:
        return next(iter(providers))
    return ""


def _preset_connector_providers() -> Dict[str, str]:
    """Read bounded, bundled manifest metadata; never infer from a path name."""
    raw_root = os.environ.get("ZETTLAB_PRESETS_DIR", "").strip()
    if not raw_root:
        return {}
    root = Path(raw_root).expanduser()
    skills_root = root / "skills"
    if not skills_root.is_dir():
        return {}
    result: Dict[str, str] = {}
    for index, manifest in enumerate(skills_root.glob("**/manifest.yaml")):
        if index >= _MAX_PRESET_MANIFESTS:
            break
        try:
            skill_id, provider_id = _connector_provider_from_manifest(manifest)
        except (OSError, UnicodeError):
            continue
        if skill_id and provider_id:
            result[skill_id] = provider_id
    return result


def _connector_provider_from_manifest(path: Path) -> tuple[str, str]:
    skill_id = ""
    provider_id = ""
    in_action_manifest = False
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if raw_line and not raw_line[0].isspace():
            in_action_manifest = raw_line.strip() == "connector_action_manifest:"
        line = raw_line.strip()
        if line.startswith("id:") and not skill_id:
            skill_id = line.partition(":")[2].strip().strip('"\'').lower()
        elif in_action_manifest and line.startswith("provider_id:"):
            candidate = line.partition(":")[2].strip().strip('"\'').lower()
            if _PROVIDER_ID_RE.fullmatch(candidate):
                provider_id = candidate
    return skill_id, provider_id


def requires_live_chat_grant(_skills: Any) -> bool:
    """Cron creation never needs a live Chat route in the direct model."""
    return False


def prepare_route_capability(job_id: str, execution_id: str, provider_id: str) -> str:
    """Ask the local-server broker for one AC-local route handle.

    ``provider_id`` may be empty: that requests an Agent-level route covering
    every Connector the Agent is authorized for, with the provider resolved
    per request by the broker. A non-empty value keeps the provider-bound
    route and must be a canonical provider id.
    """
    provider_id = str(provider_id or "").strip().lower()
    if not enabled() or (provider_id and not _PROVIDER_ID_RE.fullmatch(provider_id)):
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    result = _bridge_call(
        _PREPARE_METHOD,
        {"job_id": str(job_id), "execution_id": str(execution_id), "provider_id": provider_id},
        retry_transient=True,
    )
    capability = str(result.get("route_capability") or "").strip()
    if not capability:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    return capability


def _secret(name: str) -> str:
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        if scope is not None:
            value = scope.get(name)
            return "" if value is None else str(value)
        if is_multiplex_active():
            return ""
    except Exception:
        pass
    return os.environ.get(name, "")


def _bridge_call(method: str, params: Dict[str, Any], *, retry_transient: bool) -> Dict[str, Any]:
    url = _secret("ZETTLAB_CONNECTORS_URL").strip()
    # This is the generic local Agent-process identity already required by the
    # gateway. It is not a Connector credential, is never relayed to Server,
    # and cannot authorize a provider call by itself.
    action_token = (_secret("ZETTLAB_AGENT_ACTION_TOKEN") or _secret("ZET_AGENT_KEY")).strip()
    if not url or not action_token:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    headers = {
        "Content-Type": "application/json",
        "X-Zettlab-Agent-Action-Token": action_token,
        "X-Zettlab-Connector-Consumer": "skill-runtime",
    }
    body = json.dumps({"jsonrpc": "2.0", "id": "cron-connector-execution", "method": method, "params": params}).encode()
    attempts = 2 if retry_transient else 1
    last_error: Optional[ConnectorExecutionLeaseError] = None
    for attempt in range(attempts):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode() or "{}")
        except HTTPError as exc:
            last_error = ConnectorExecutionLeaseError("task_connector_temporarily_unavailable", retryable=exc.code >= 500)
        except (URLError, TimeoutError, OSError):
            last_error = ConnectorExecutionLeaseError("task_connector_temporarily_unavailable", retryable=True)
        else:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                data = error.get("data")
                code = data.get("error_code") if isinstance(data, dict) else ""
                raise ConnectorExecutionLeaseError(str(code or error.get("message") or "task_connector_not_authorized"))
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, dict):
                return result
            raise ConnectorExecutionLeaseError("task_connector_not_authorized")
        if last_error and (not last_error.retryable or attempt + 1 == attempts):
            raise last_error
        time.sleep(0.15 * (attempt + 1))
    raise last_error or ConnectorExecutionLeaseError("task_connector_temporarily_unavailable")
