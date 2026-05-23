#!/usr/bin/env python3
"""Zettlab connector tools derived from registered skill manifests.

Zettlab installs connector-backed skills into the active Hermes profile. Those
skills declare runtime entry points in SKILL.md, usually under
``prerequisites.tools``. This module turns those declared entry points into
native Hermes tools and proxies calls to the current Zettlab connector runtime.

It intentionally does not synthesize providers or return fake success. If the
runtime URL, auth token, server tool, or provider authorization is missing, the
real connector call fails and the model receives that error.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List

from hermes_cli.config import cfg_get, read_raw_config
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_CONNECTOR_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9_.-]+$")
_MODEL_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_TOOLSET = "zettlab-connectors"
_REQUEST_TIMEOUT_SECONDS = 120
_DEFAULT_DISPATCHER_TOOLS = (
    "custom_connector.list_tools",
    "custom_connector.call_tool",
)


def _is_connector_tool_name(name: str) -> bool:
    return bool(_CONNECTOR_TOOL_RE.match(str(name).strip()))


def _model_tool_name(canonical_name: str) -> str:
    return _MODEL_TOOL_NAME_RE.sub("_", str(canonical_name).strip())


def _normalise_tool_schema(schema: Any) -> Dict[str, Any]:
    """Return an OpenAI-compatible object schema for a connector tool.

    The connector runtime speaks JSON Schema, but some providers omit
    optional fields as null. OpenAI-compatible function calling rejects
    ``required: null``; normalize that at the bridge so canonical connector
    schemas can pass through without leaking provider quirks into Hermes.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, child in value.items():
                if key == "required":
                    if isinstance(child, list):
                        out[key] = [item for item in child if isinstance(item, str)]
                    elif child is not None:
                        out[key] = []
                    continue
                out[key] = clean(child)
            return out
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    normalized = clean(copy.deepcopy(schema))
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    normalized.setdefault("type", "object")
    if normalized.get("type") == "object" and not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    return normalized


def _normalise_names(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _is_zettlab_connector_skill(frontmatter: Dict[str, Any]) -> bool:
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return False
    zettlab = metadata.get("zettlab")
    if not isinstance(zettlab, dict):
        return False
    return bool(zettlab.get("connector_skill"))


def _connector_tools_from_frontmatter(frontmatter: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()

    def add(values: Any) -> None:
        for name in _normalise_names(values):
            if not _is_connector_tool_name(name) or name in seen:
                continue
            seen.add(name)
            names.append(name)

    prerequisites = frontmatter.get("prerequisites")
    if isinstance(prerequisites, dict):
        add(prerequisites.get("tools"))
    add(frontmatter.get("required_tools"))

    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        zettlab = metadata.get("zettlab")
        if isinstance(zettlab, dict):
            add(zettlab.get("connectorTools"))
            add(zettlab.get("connector_tools"))

    return names


def _registered_skill_connector_tools() -> List[str]:
    names: List[str] = list(_DEFAULT_DISPATCHER_TOOLS)
    seen: set[str] = set(names)
    try:
        from agent.skill_utils import get_all_skills_dirs, iter_skill_index_files
        from tools.skills_tool import (
            _EXCLUDED_SKILL_DIRS,
            _is_skill_disabled,
            _parse_frontmatter,
            skill_matches_platform,
        )
    except Exception as exc:
        logger.debug("Zettlab connector tools: cannot inspect skills: %s", exc)
        return names

    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_md in iter_skill_index_files(skills_dir, "SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, _ = _parse_frontmatter(content)
            except Exception as exc:
                logger.debug("Zettlab connector tools: skip %s: %s", skill_md, exc)
                continue
            if not _is_zettlab_connector_skill(frontmatter):
                continue
            if not skill_matches_platform(frontmatter):
                continue
            skill_name = str(frontmatter.get("name") or skill_md.parent.name).strip()
            if skill_name and _is_skill_disabled(skill_name):
                continue
            for tool_name in _connector_tools_from_frontmatter(frontmatter):
                if tool_name in seen:
                    continue
                seen.add(tool_name)
                names.append(tool_name)
    return names


def _connector_url() -> str:
    raw = (
        os.getenv("ZETTLAB_CONNECTORS_URL")
        or os.getenv("ZETTLAB_CONNECTORS_RPC_URL")
        or os.getenv("ZETTLAB_CONNECTORS_MCP_URL")
        or ""
    ).strip()
    if not raw:
        cfg = read_raw_config()
        raw = str(
            cfg_get(
                cfg,
                "mcp_servers",
                "zettlab_connectors",
                "url",
                default="",
            )
            or ""
        ).strip()
    return os.path.expandvars(raw)


def _connector_token() -> str:
    return os.getenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "").strip()


def _json_rpc(method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = _connector_url()
    if not url:
        raise RuntimeError("zettlab_connector_runtime_url_missing")
    token = _connector_token()
    if not token:
        raise RuntimeError("zettlab_connector_auth_token_missing")

    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "zettlab-connector-tools",
            "method": method,
            "params": params or {},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Zettlab-Client": "hermes-agent",
        "X-Zettlab-Tool-Bridge": "skill-tools",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"zettlab_connector_http_{exc.code}:{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"zettlab_connector_unreachable:{exc.reason}") from exc

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("zettlab_connector_invalid_json") from exc
    if isinstance(decoded, dict) and decoded.get("error"):
        error = decoded["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise ConnectorRPCError(message or "zettlab_connector_error", error)
    if not isinstance(decoded, dict):
        raise RuntimeError("zettlab_connector_invalid_response")
    return decoded


class ConnectorRPCError(RuntimeError):
    def __init__(self, message: str, payload: Any):
        super().__init__(message)
        self.payload = payload


def _list_available_connector_schemas(tool_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    wanted = set(tool_names)
    if not wanted:
        return {}
    try:
        result = _json_rpc("tools/list").get("result") or {}
    except Exception as exc:
        logger.debug("Zettlab connector tools: tools/list unavailable: %s", exc)
        return {}

    schemas: Dict[str, Dict[str, Any]] = {}
    for item in result.get("tools") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name not in wanted:
            continue
        schema = _normalise_tool_schema(item.get("inputSchema"))
        alias = _model_tool_name(name)
        description = str(item.get("description") or f"Call Zettlab connector tool {name}.")
        schemas[name] = {
            "name": alias,
            "description": (
                f"Internal model-safe alias for connector tool `{name}`. "
                f"When naming this tool in user-facing text, use `{name}`. {description}"
            ),
            "parameters": schema,
        }
    return schemas


def _generic_tool_schema(name: str) -> Dict[str, Any]:
    provider = name.split(".", 1)[0]
    return {
        "name": _model_tool_name(name),
        "description": (
            f"Internal model-safe alias for connector tool `{name}`. "
            f"When naming this tool in user-facing text, use `{name}`. "
            f"Call the Zettlab {provider} connector runtime with the current "
            "user, agent, and chat authorization context."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    }


def _call_connector_tool(name: str, args: Dict[str, Any] | None = None, **_: Any) -> str:
    try:
        response = _json_rpc("tools/call", {"name": name, "arguments": args or {}})
    except ConnectorRPCError as exc:
        return tool_error(str(exc), connector_error=exc.payload)
    except Exception as exc:
        return tool_error(str(exc), errorCode="connector_runtime_error")
    result = response.get("result")
    if isinstance(result, dict):
        return tool_result(result)
    return tool_result({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})


def _register_skill_declared_connector_tools() -> List[str]:
    tool_names = _registered_skill_connector_tools()
    hydrated = _list_available_connector_schemas(tool_names)
    registered: List[str] = []
    aliases_seen: set[str] = set()
    for tool_name in tool_names:
        model_name = _model_tool_name(tool_name)
        if not model_name or model_name in aliases_seen:
            logger.warning(
                "Zettlab connector tools skipped duplicate model alias %s for %s",
                model_name,
                tool_name,
            )
            continue
        aliases_seen.add(model_name)
        schema = hydrated.get(tool_name) or _generic_tool_schema(tool_name)
        registry.register(
            name=model_name,
            toolset=_TOOLSET,
            schema=schema,
            handler=lambda args, _name=tool_name, **kw: _call_connector_tool(_name, args, **kw),
            description=schema.get("description", ""),
            max_result_size_chars=20000,
        )
        registered.append(model_name)
    if registered:
        logger.info("Zettlab connector tools registered: %s", ", ".join(registered))
    return registered


def _bridge_status_tool(**_: Any) -> str:
    return tool_result(
        {
            "url_configured": bool(_connector_url()),
            "auth_token_configured": bool(_connector_token()),
            "registered_tools": _registered_skill_connector_tools(),
        }
    )


registry.register(
    name="zettlab_connector_bridge_status",
    toolset=_TOOLSET,
    schema={
        "name": "zettlab_connector_bridge_status",
        "description": "Internal Zettlab connector runtime readiness probe.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    handler=lambda args, **kw: _bridge_status_tool(**kw),
    check_fn=lambda: False,
    description="Internal Zettlab connector runtime readiness probe.",
)

_register_skill_declared_connector_tools()
