"""Cron-only Connector execution lease client.

The durable grant is scoped to one job and can only be exchanged by the
device-authenticated local bridge. A cron run receives only a local route
capability; neither a user JWT nor the short cloud lease reaches Hermes tools.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_CREATE_METHOD = "zettlab/cron/create-connector-grant"
_LEASE_METHOD = "zettlab/cron/acquire-connector-lease"
_TIMEOUT_SECONDS = 10


class ConnectorExecutionLeaseError(RuntimeError):
    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.code = code or "task_connector_temporarily_unavailable"
        self.retryable = retryable


def enabled() -> bool:
    return _secret("ZETTLAB_CRON_CONNECTOR_EXECUTION_LEASES_ENABLED") == "1"


def wants_linear_execution(skills: Any) -> bool:
    # Legacy jobs can retain the singular ``skill: "Linear"`` shape.  Treat
    # it as one value instead of iterating its characters and accidentally
    # bypassing the task-authorization requirement.
    if isinstance(skills, str):
        skills = [skills]
    return any(str(skill).strip().lower() == "linear" for skill in (skills or []))


def supports_exclusive_linear_execution(skills: Any) -> bool:
    """Whether one task can safely use the single-connection Linear lease."""
    if isinstance(skills, str):
        skills = [skills]
    normalized = [str(skill).strip().lower() for skill in (skills or []) if str(skill).strip()]
    return bool(normalized) and all(skill == "linear" for skill in normalized)


def requires_live_chat_grant(skills: Any) -> bool:
    """Whether creating this job needs the live Chat-only grant exchange."""
    return enabled() and wants_linear_execution(skills)


def create_grant(job_id: str, provider_id: str = "linear") -> Dict[str, str]:
    if not enabled() or provider_id.strip().lower() != "linear":
        return {}
    try:
        from gateway.session_context import zettlab_connector_route_capability
        route_capability = zettlab_connector_route_capability()
    except Exception:
        route_capability = ""
    if not route_capability:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    result = _bridge_call(
        _CREATE_METHOD,
        {"job_id": str(job_id), "provider_id": "linear"},
        route_capability=route_capability,
        retry_transient=False,
    )
    grant_id = str(result.get("grant_id") or "").strip()
    grant_token = str(result.get("grant_token") or "").strip()
    if not grant_id or not grant_token:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    return {
        "provider_id": "linear",
        "grant_id": grant_id,
        "grant_token": grant_token,
        "expires_at": str(result.get("expires_at") or "").strip(),
    }


def acquire_route_capability(execution: Any, execution_id: str) -> str:
    if not isinstance(execution, dict):
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    if str(execution.get("provider_id") or "").strip().lower() != "linear":
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    grant_token = str(execution.get("grant_token") or "").strip()
    if not grant_token:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    result = _bridge_call(
        _LEASE_METHOD,
        {"grant_token": grant_token, "execution_id": str(execution_id)},
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


def _bridge_call(method: str, params: Dict[str, Any], *, route_capability: str = "", retry_transient: bool) -> Dict[str, Any]:
    url = _secret("ZETTLAB_CONNECTORS_URL").strip()
    action_token = _secret("ZETTLAB_CONNECTORS_AUTH_TOKEN").strip()
    if not url or not action_token:
        raise ConnectorExecutionLeaseError("task_connector_not_authorized")
    headers = {
        "Content-Type": "application/json",
        "X-Zettlab-Agent-Action-Token": action_token,
        "X-Zettlab-Connector-Consumer": "skill-runtime",
    }
    if route_capability:
        headers["X-Zettlab-Session-Key"] = route_capability
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
