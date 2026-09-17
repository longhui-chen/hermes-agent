"""Current-session Connector creation through the local broker, without shell argv."""

import json
import re
import urllib.request
import uuid
from urllib.parse import urlsplit

from tools.registry import registry


_CREATE_TOOL = "connector.create_readonly_template"
_TEMPLATE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_KEY_RE = re.compile(r"[A-Za-z0-9._:-]{16,128}\Z")


def _runtime_env() -> dict[str, str]:
    # Reuse the existing profile-scoped allowlist; never inherit another
    # profile's process environment or give provider credentials to a shell.
    from tools.environments.local import build_connector_runtime_env

    return build_connector_runtime_env()


def _validated_env() -> dict[str, str] | None:
    env = _runtime_env()
    url = str(env.get("ZETTLAB_CONNECTORS_URL") or "")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path != "/api/v1/internal/connectors/rpc"):
        return None
    for key in ("ZETTLAB_AGENT_ACTION_TOKEN", "ZETTLAB_CONNECTOR_SESSION_ID", "ZETTLAB_TURN_ID"):
        value = str(env.get(key) or "").strip()
        if not value or len(value) > 128 or any(char in value for char in "\r\n\x00"):
            return None
    return env


def _check_connector_chat_create() -> bool:
    return _validated_env() is not None


_check_connector_chat_create._profile_scope_sensitive = True  # type: ignore[attr-defined]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _rpc(request: urllib.request.Request, timeout: float) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(64 << 10)
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _call(env: dict[str, str], method: str, params: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        str(env["ZETTLAB_CONNECTORS_URL"]),
        data=json.dumps({"jsonrpc": "2.0", "id": uuid.uuid4().hex,
                         "method": method, "params": params}).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json", "Accept": "application/json",
            "X-Zettlab-Agent-Action-Token": str(env["ZETTLAB_AGENT_ACTION_TOKEN"]),
            "X-Zettlab-Connector-Session-Id": str(env["ZETTLAB_CONNECTOR_SESSION_ID"]),
            "X-Zettlab-Turn-Id": str(env["ZETTLAB_TURN_ID"]),
            "X-Zettlab-Connector-Consumer": "skill-runtime",
            "X-Zettlab-Connector-Session-Invoke": "v1",
        },
    )
    return _rpc(request, timeout)


def _safe_error(response: dict) -> str:
    error = response.get("error")
    code = error.get("message") if isinstance(error, dict) else ""
    if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{2,90}", code):
        return code
    return "connector_chat_create_unavailable"


def connector_chat_create_tool(args, **kw) -> str:
    env = _validated_env()
    action = args.get("action") if isinstance(args, dict) else None
    if env is None or action not in {"probe", "create"}:
        return json.dumps({"ok": False, "available": False, "code": "connector_chat_create_unavailable"})
    if action == "probe":
        try:
            response = _call(env, "tools/list", {}, 5.0)
            tools = (response.get("result") or {}).get("tools")
            available = isinstance(tools, list) and any(
                isinstance(tool, dict) and tool.get("name") == _CREATE_TOOL for tool in tools
            )
            return json.dumps({"ok": True, "available": available})
        except Exception:
            return json.dumps({"ok": False, "available": False, "code": "connector_chat_create_unavailable"})

    template = args.get("template_id")
    variables = args.get("variables")
    secret = args.get("secret")
    key = args.get("idempotency_key")
    if (not isinstance(template, str) or len(template) > 128
            or _TEMPLATE_ID_RE.fullmatch(template) is None
            or args.get("template_version") != 1
            or not isinstance(variables, dict) or len(variables) > 16
            or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 512
                   for k, v in variables.items())
            or not isinstance(secret, str) or not secret.strip() or len(secret) > 4096
            or not isinstance(key, str) or _KEY_RE.fullmatch(key) is None
            or not isinstance(args.get("insecure_transport_confirmed", False), bool)):
        return json.dumps({"ok": False, "code": "connector_chat_create_invalid_request"})
    arguments = {name: args[name] for name in (
        "template_id", "template_version", "variables", "secret", "idempotency_key",
    )}
    arguments["insecure_transport_confirmed"] = args.get("insecure_transport_confirmed", False)
    try:
        response = _call(env, "tools/call", {"name": _CREATE_TOOL, "arguments": arguments}, 30.0)
    except Exception:
        # Unknown create outcome: the next turn must query/replay the same key.
        return json.dumps({"ok": False, "code": "connector_chat_create_outcome_unknown"})
    if response.get("error"):
        return json.dumps({"ok": False, "code": _safe_error(response)})
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("connection_id"), str):
        return json.dumps({"ok": False, "code": "connector_chat_create_invalid_result"})
    return json.dumps({
        "ok": True, "connection_id": result["connection_id"],
        "connection_status": str(result.get("connection_status") or ""),
        "template_id": str(result.get("template_id") or ""),
        "session_ready": result.get("session_ready") is True,
        "verified": result.get("verified") is True,
    })


registry.register(
    name="connector_chat_create", toolset="zettlab_connectors",
    schema={
        "name": "connector_chat_create",
        "description": "Probe current-Chat Connector creation availability, then create and enable one published Connector template with its declared tools available next turn. Write and destructive tools retain policy and confirmation gates. No shell.",
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["probe", "create"]},
                "template_id": {"type": "string", "minLength": 1, "maxLength": 128,
                                "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
                "template_version": {"type": "integer"},
                "variables": {"type": "object", "additionalProperties": {"type": "string"}},
                "secret": {"type": "string"},
                "insecure_transport_confirmed": {"type": "boolean"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["action"],
        },
    },
    handler=connector_chat_create_tool,
    check_fn=_check_connector_chat_create,
    emoji="🔗",
    max_result_size_chars=1000,
    defer_to_tool_search=False,
)
