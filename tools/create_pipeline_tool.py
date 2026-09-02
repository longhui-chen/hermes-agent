"""Trigger the platform's application-creation pipeline (start + poll status).

app-dispatch routes a NEW-app request here instead of the main Chat Agent
free-lancing skills (product-prototyping, app-coding, …): the platform pipeline
runs guide→assemble→coding→compile→selftest→publish with the order welded in, so
no stray skill can intercept creation and no step is skipped. Loopback-only, same
trust model as app_host. The run executes in the background — `create` returns a
run_id immediately; poll `status` until done/failed.

Registered under the zettlab_apphost toolset (verified on-device: a separate
toolset's tool was silently dropped by the platform reverse-mapping; app_host's
toolset resolves + carries the same App Host secret scope).
"""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from tools.registry import registry

_ACTIONS = {"create", "status"}
_SCHEMA = {
    "name": "create_pipeline",
    "description": (
        "Start the platform's application-creation pipeline from a one-line intent, "
        "or poll a run. Use this for a NEW app request INSTEAD of writing code or "
        "opening prototyping skills yourself — the platform drives "
        "guide->assemble->coding->compile->selftest->publish deterministically, in order. "
        "action='create' needs `intent` (the user's one-line request) and returns a "
        "run_id; action='status' needs `run_id` and returns step/done/failed/entry_url."
    ),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "intent": {"type": "string", "maxLength": 4096},
        "run_id": {"type": "string", "maxLength": 80},
    }, "required": ["action"], "additionalProperties": False},
}


def _secret(name: str) -> str:
    return str(get_secret(name, "") or "").strip()


def _base_url() -> str:
    """Derive the create-pipeline loopback base from the app_host base
    (…/api/v1/internal/apphost -> …/api/v1/internal/createpipeline). Reuses the
    already-injected ZET_APPHOST_BASE_URL so no new registry env is needed."""
    raw = _secret("ZET_APPHOST_BASE_URL")
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or not parsed.port or not ipaddress.ip_address(host).is_loopback:
            return ""
    except (ValueError, TypeError):
        return ""
    path = parsed.path.rstrip("/")
    if path.endswith("/apphost"):
        path = path[: -len("/apphost")] + "/createpipeline"
    elif "/internal" in path:
        path = path.rsplit("/internal", 1)[0] + "/internal/createpipeline"
    else:
        path = path + "/createpipeline"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _cp_enabled() -> bool:
    # Gate on the App Host base secret + the action token (same session condition
    # as app_host). The /createpipeline sibling is always reachable when /apphost
    # is; do NOT gate on the derived URL (call-time concern). A unique name avoids
    # any registry check_fn keying collision with other tools' `_enabled`.
    return bool(_secret("ZET_APPHOST_BASE_URL") and _secret("ZETTLAB_AGENT_ACTION_TOKEN"))


_cp_enabled._profile_scope_sensitive = True  # type: ignore[attr-defined]


def create_pipeline(args: Any = None, **_: Any) -> str:
    # hermes dispatch calls handler(args, **kwargs): the first positional `args`
    # is the parsed argument dict (see app_host_tool's args.get(...) pattern —
    # the unpacked-kwargs signature crashes with "unhashable type: 'dict'").
    if not isinstance(args, dict):
        args = {}
    action = str(args.get("action", "") or "").strip()
    if action not in _ACTIONS:
        return json.dumps({"success": False, "code": "invalid_action"}, ensure_ascii=False)
    base = _base_url()
    if not base:
        return json.dumps({"success": False, "code": "create_pipeline_unavailable"}, ensure_ascii=False)
    headers = {"X-Zettlab-Agent-Action-Token": _secret("ZETTLAB_AGENT_ACTION_TOKEN")}
    try:
        if action == "create":
            intent = str(args.get("intent", "") or "").strip()
            if not intent:
                return json.dumps({"success": False, "code": "intent_required"}, ensure_ascii=False)
            body = {"intent": intent}
            owner = _secret("ZET_AGENT_ID")
            if owner:
                body["owner_agent"] = owner
            response = requests.post(base + "/create", headers=headers, json=body, timeout=15)
        else:  # status
            run_id = str(args.get("run_id", "") or "").strip()
            if not run_id:
                return json.dumps({"success": False, "code": "run_id_required"}, ensure_ascii=False)
            response = requests.get(base + "/status/" + quote(run_id, safe=""), headers=headers, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False)
    except (requests.RequestException, ValueError) as exc:
        return json.dumps({"success": False, "code": "create_pipeline_unavailable", "error": str(exc)[:256]}, ensure_ascii=False)


registry.register(name="create_pipeline", toolset="zettlab_apphost", schema=_SCHEMA, handler=create_pipeline, check_fn=_cp_enabled, defer_to_tool_search=False)
