"""Owner-scoped bridge to declared Generated App operations.

Hermes deliberately knows nothing about application-specific operation names or
payload schemas.  local-server derives the operation catalog from the owning
app's metadata, validates the app owner and operation schema, and dispatches the
fixed internal route.  This module only owns the model-facing transport
boundary: fixed loopback routing, bounded JSON, sensitive-field rejection,
mutation approval, and retry policy.

Application responses are untrusted.  They are bounded and explicitly marked;
this transport never interprets response content as instructions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from agent.secret_scope import get_secret
from tools.loopback_transport import (
    HARDENED_OPENER as _NO_PROXY_OPENER,
    is_trusted_loopback_http as _is_trusted_loopback_http,
)
from tools.registry import registry


logger = logging.getLogger(__name__)

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_BASE_URL_SECRET = "ZET_APPHOST_BASE_URL"
_ACTION_TOKEN_SECRET = "ZETTLAB_AGENT_ACTION_TOKEN"
_AGENT_ID_SECRET = "ZET_AGENT_ID"
_CAPABILITY_TIMEOUT = 8.0
_MUTATION_TIMEOUT = 25.0
_READ_TIMEOUT = 35.0
_MAX_RESPONSE_BYTES = 512 * 1024
_MAX_PAYLOAD_BYTES = 128 * 1024
_MAX_ENVELOPE_BYTES = 144 * 1024
_MAX_DOCUMENT_DEPTH = 16
_MAX_DOCUMENT_NODES = 4096
_MAX_KEY_CHARS = 128
_MAX_STRING_CHARS = 64 * 1024
_MAX_QUERY_ITEMS = 16
_MAX_QUERY_VALUE_CHARS = 1024
_MAX_ERROR_MESSAGE_CHARS = 320
_MAX_OPERATIONS = 128
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{2,31}$")
_OPERATION_RE = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,7}$"
)
_QUERY_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CAPABILITY_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIONS = frozenset({"capabilities", "invoke"})
_REQUEST_KEYS = frozenset(
    {"action", "slug", "operation", "query", "payload", "idempotency_key"}
)
_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "agentid",
        "accesskey",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "deliverytarget",
        "deliverto",
        "destination",
        "jwt",
        "password",
        "secret",
        "secretkey",
    }
)
_FORBIDDEN_KEY_MARKERS = (
    "token",
    "header",
    "method",
    "prompt",
    "provider",
    "model",
    "skill",
)


class _BadRequest(ValueError):
    """A model-facing validation failure; no operation has been invoked."""


@dataclass(frozen=True)
class _BridgeError(Exception):
    code: str
    message: str
    status: int | None


def _secret(name: str) -> str:
    return str(get_secret(name, "") or "").strip()


def _urlopen(request: urllib.request.Request, timeout: float):
    return _NO_PROXY_OPENER.open(request, timeout=timeout)


def _base_url() -> str | None:
    raw = _secret(_BASE_URL_SECRET)
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.path.rstrip("/") != "/api/v1/internal/apps":
        return None
    if not parts.netloc or not _is_trusted_loopback_http(parts):
        return None
    return raw.rstrip("/")


def _is_cron_session() -> bool:
    try:
        from gateway.session_context import get_session_env
        from utils import is_truthy_value

        return is_truthy_value(get_session_env("HERMES_CRON_SESSION", ""))
    except Exception:
        # This is a scope boundary: an unavailable session-context lookup must
        # never turn an unknown Cron request into an allowed App Data call.
        logger.warning("Unable to resolve Cron session scope; denying App Data")
        return True


def _is_delegated_child_context() -> bool:
    """Return whether this tool is running in a delegated child scope.

    Child agents inherit the parent's profile secret scope, so an unavailable
    context lookup is treated as child scope rather than granting access.
    """
    try:
        from agent.delegation_context import is_delegated_child_context

        return bool(is_delegated_child_context())
    except Exception:
        logger.warning(
            "Unable to resolve delegated-child scope; denying App Data"
        )
        return True


def _is_trusted_zet_agent_session() -> bool:
    """Return whether the task-local host is the Zet Agent API surface."""
    try:
        from gateway.session_context import get_session_env

        return (
            str(get_session_env("HERMES_SESSION_PLATFORM", "") or "")
            .strip()
            .lower()
            == "zet_agent"
        )
    except Exception:
        logger.warning(
            "Unable to resolve App Data session platform; denying access"
        )
        return False


def _check_app_data() -> bool:
    if (
        _is_cron_session()
        or _is_delegated_child_context()
        or not _is_trusted_zet_agent_session()
    ):
        return False
    return bool(
        _base_url()
        and _secret(_ACTION_TOKEN_SECRET)
        and _secret(_AGENT_ID_SECRET)
    )


_check_app_data._profile_scope_sensitive = True  # type: ignore[attr-defined]
_check_app_data._session_scope_sensitive = True  # type: ignore[attr-defined]


def _failure(code: str, message: str, *, status: int | None) -> str:
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message}, "status": status},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _success(data: object) -> str:
    return json.dumps(
        {"ok": True, "data": data, "untrusted_app_data": True},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _safe_error(raw: bytes) -> dict[str, str] | None:
    if len(raw) > 16 * 1024:
        return None
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    code = value.get("code")
    message = value.get("message")
    if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,96}", code):
        return None
    if not isinstance(message, str):
        message = "应用数据操作失败"
    message = "".join(ch for ch in message if ch >= " " or ch in "\t\n")
    return {"code": code, "message": message[:_MAX_ERROR_MESSAGE_CHARS]}


def _validate_slug(value: object) -> str:
    if not isinstance(value, str):
        raise _BadRequest("slug 必须是字符串")
    slug = value.strip()
    if not _SLUG_RE.fullmatch(slug):
        raise _BadRequest("slug 格式不合法")
    return slug


def _validate_action(value: object) -> str:
    if not isinstance(value, str) or value not in _ACTIONS:
        raise _BadRequest("action 必须是 capabilities 或 invoke")
    return value


def _validate_operation(value: object) -> str:
    if not isinstance(value, str) or not _OPERATION_RE.fullmatch(value.strip()):
        raise _BadRequest("operation 格式不合法")
    return value.strip()


def _is_forbidden_key(key: str) -> bool:
    # Match Local Server's canonical field rule so separator/camelCase variants
    # converge. Only the fixed source_url record field may carry a URL.
    canonical = "".join(char for char in key.lower() if char.isalnum())
    if canonical == "sourceurl":
        return False
    if canonical in _FORBIDDEN_EXACT_KEYS:
        return True
    if any(marker in canonical for marker in _FORBIDDEN_KEY_MARKERS):
        return True
    return canonical.endswith(("url", "uri", "path"))


def _validate_document(name: str, raw: object) -> dict[str, object] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _BadRequest(f"{name} 必须是 JSON 对象")

    nodes = 0
    stack: list[tuple[object, int]] = [(raw, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_DOCUMENT_NODES:
            raise _BadRequest(f"{name} 结构过大")
        if depth > _MAX_DOCUMENT_DEPTH:
            raise _BadRequest(f"{name} 嵌套过深")
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > _MAX_KEY_CHARS:
                    raise _BadRequest(f"{name} 字段名不合法")
                if any(ord(char) < 0x20 or ord(char) == 0x7F for char in key):
                    raise _BadRequest(f"{name} 字段名含控制字符")
                if _is_forbidden_key(key):
                    raise _BadRequest(f"{name} 含受保护字段")
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            if len(value) > _MAX_STRING_CHARS:
                raise _BadRequest(f"{name} 字符串过长")
            if any(ord(char) < 0x20 and char not in "\t\n\r" for char in value):
                raise _BadRequest(f"{name} 字符串含控制字符")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise _BadRequest(f"{name} 含非 JSON 值")

    try:
        encoded = json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise _BadRequest(f"{name} 含非 JSON 值") from exc
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise _BadRequest(f"{name} 过大")
    return dict(raw)


def _validate_query(raw: object) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _BadRequest("query 必须是 JSON 对象")
    if len(raw) > _MAX_QUERY_ITEMS:
        raise _BadRequest("query 字段过多")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _QUERY_KEY_RE.fullmatch(key):
            raise _BadRequest("query 字段名不合法")
        if _is_forbidden_key(key):
            raise _BadRequest("query 含受保护字段")
        if not isinstance(value, str) or len(value) > _MAX_QUERY_VALUE_CHARS:
            raise _BadRequest("query 值不合法")
        if any(ord(char) < 0x20 and char not in "\t\n\r" for char in value):
            raise _BadRequest("query 值含控制字符")
        result[key] = value
    return result


def _validate_key(value: object) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _IDEMPOTENCY_RE.fullmatch(value.strip()):
        raise _BadRequest("idempotency_key 格式不合法")
    return value.strip()


def _profile_scope_digest() -> str:
    identity = _secret(_AGENT_ID_SECRET)
    if not identity:
        raise _BadRequest("当前 Agent 缺少 profile identity")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _parse_capabilities(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value.get("version") != 1
    ):
        raise ValueError("invalid capability response")
    operations = value.get("operations")
    capability_digest = value.get("capability_digest")
    if not isinstance(operations, list) or len(operations) > _MAX_OPERATIONS:
        raise ValueError("invalid capability response")
    if (
        not isinstance(capability_digest, str)
        or _CAPABILITY_DIGEST_RE.fullmatch(capability_digest) is None
    ):
        raise ValueError("invalid capability response")

    clean: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in operations:
        if not isinstance(item, dict) or "name" not in item or "mode" not in item:
            raise ValueError("invalid capability response")
        name = item.get("name")
        mode = item.get("mode")
        if not isinstance(name, str) or not _OPERATION_RE.fullmatch(name):
            raise ValueError("invalid capability response")
        if (
            not isinstance(mode, str)
            or mode not in {"read", "mutation"}
            or name in seen
        ):
            raise ValueError("invalid capability response")
        seen.add(name)
        clean.append({"name": name, "mode": mode})
    return {
        "version": 1,
        "operations": clean,
        "capability_digest": capability_digest,
    }


def _public_capabilities(capabilities: dict[str, object]) -> dict[str, object]:
    """Project declarations without exposing the transport-only CAS token."""
    return {
        "version": capabilities["version"],
        "operations": capabilities["operations"],
    }


def _bridge_error(exc: _BridgeError) -> str:
    return _failure(exc.code, exc.message, status=exc.status)


def _request_json(
    *,
    base: str,
    token: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
    retry_read: bool,
    timeout: float,
) -> object:
    encoded: bytes | None = None
    if body is not None:
        try:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise _BridgeError("invalid_request", "请求包含无效 JSON", 0) from exc
        if len(encoded) > _MAX_ENVELOPE_BYTES:
            raise _BridgeError("invalid_request", "请求体过大", 0)

    attempts = 2 if retry_read else 1
    for attempt in range(attempts):
        request = urllib.request.Request(
            base + path,
            data=encoded,
            headers={
                _ACTION_TOKEN_HEADER: token,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with _urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(_MAX_RESPONSE_BYTES + 1) or b""
            if retry_read and exc.code in {429, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(0.05)
                continue
            upstream = _safe_error(raw)
            if upstream is not None:
                raise _BridgeError(upstream["code"], upstream["message"], exc.code) from exc
            raise _BridgeError(
                "transport_error",
                "本地 App Data 数据桥返回了无法解析的错误",
                exc.code,
            ) from exc
        except Exception as exc:
            if retry_read and attempt + 1 < attempts:
                time.sleep(0.05)
                continue
            raise _BridgeError(
                "transport_error",
                "无法连接本地 App Data 数据桥",
                None,
            ) from exc

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise _BridgeError("response_too_large", "App Data 响应超过大小上限", status)
        if status < 200 or status >= 300:
            upstream = _safe_error(raw)
            if upstream is not None:
                raise _BridgeError(upstream["code"], upstream["message"], status)
            raise _BridgeError("transport_error", "App Data 操作失败", status)
        try:
            return json.loads(raw.decode("utf-8", errors="replace")) if raw.strip() else {}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _BridgeError(
                "invalid_response",
                "本地 App Data 数据桥返回了无效 JSON",
                status,
            ) from exc

    raise _BridgeError("transport_error", "App Data 操作失败", None)


def _load_capabilities(base: str, token: str, slug: str) -> dict[str, object]:
    path = f"/{quote(slug, safe='')}/capabilities"
    value = _request_json(
        base=base,
        token=token,
        method="GET",
        path=path,
        body=None,
        retry_read=True,
        timeout=_CAPABILITY_TIMEOUT,
    )
    try:
        return _parse_capabilities(value)
    except ValueError as exc:
        raise _BridgeError(
            "invalid_response",
            "本地 App Data capability 声明无效",
            200,
        ) from exc


def _declared_mode(capabilities: dict[str, object], operation: str) -> str | None:
    operations = capabilities["operations"]
    assert isinstance(operations, list)
    for item in operations:
        assert isinstance(item, dict)
        if item.get("name") == operation:
            mode = item.get("mode")
            return mode if isinstance(mode, str) else None
    return None


def _approval_result(
    slug: str,
    operation: str,
    envelope: dict[str, object],
) -> dict:
    if _is_cron_session():
        return {
            "approved": False,
            "message": "Cron 运行不能替用户执行应用数据修改。",
            "status": "blocked",
        }
    canonical = json.dumps(
        {
            "profile_scope": _profile_scope_digest(),
            "slug": slug,
            "operation": operation,
            "envelope": envelope,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    display = canonical
    if len(display) > 2048:
        display = f"{display[:1800]}\n[truncated sha256={digest}]"
    from tools.approval import request_tool_approval

    return request_tool_approval(
        "app_data",
        f"{operation} 会修改此应用的数据。",
        rule_key=f"app_data:{operation}:{digest}",
        one_shot=True,
        allow_yolo_bypass=False,
        display_target=display,
    )


def _approval_failure(operation: str, approval: dict) -> str:
    pending = approval.get("status") in {"approval_required", "pending_approval"}
    code = "approval_required" if pending else "approval_denied"
    message = str(approval.get("message") or f"{operation} 未获得用户确认")[:_MAX_ERROR_MESSAGE_CHARS]
    result: dict[str, object] = {
        "ok": False,
        "error": {"code": code, "message": message},
        "status": 0,
    }
    approval_id = approval.get("approval_id")
    if pending and isinstance(approval_id, str) and len(approval_id) <= 128:
        result["approval_id"] = approval_id
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _run_app_data_tool(args) -> str:
    args = args if isinstance(args, dict) else {}
    if _is_cron_session():
        return _failure(
            "cron_scope_denied",
            "Cron 运行无权访问 App Data 数据桥",
            status=0,
        )
    if _is_delegated_child_context():
        return _failure(
            "delegated_child_scope_denied",
            "delegate_task 子 Agent 无权访问 App Data 数据桥",
            status=0,
        )
    if not _is_trusted_zet_agent_session():
        return _failure(
            "platform_scope_denied",
            "只有 Zet Agent 交互会话可以访问 App Data 数据桥",
            status=0,
        )
    try:
        if any(not isinstance(key, str) for key in args) or not set(args).issubset(
            _REQUEST_KEYS
        ):
            raise _BadRequest("请求包含未声明字段")
        slug = _validate_slug(args.get("slug"))
        action = _validate_action(args.get("action"))
        operation_raw = args.get("operation")
        payload_raw = args.get("payload")
        query_raw = args.get("query")
        key_raw = args.get("idempotency_key")
        if action == "capabilities":
            if any(value not in (None, "") for value in (operation_raw, payload_raw, query_raw, key_raw)):
                raise _BadRequest("capabilities 不接受 operation、payload、query 或 idempotency_key")
            operation = ""
            payload = None
            query = None
            key = ""
        else:
            operation = _validate_operation(operation_raw)
            payload = _validate_document("payload", payload_raw)
            query = _validate_query(query_raw)
            key = _validate_key(key_raw)
    except _BadRequest as exc:
        return _failure("invalid_request", str(exc), status=0)

    base = _base_url()
    token = _secret(_ACTION_TOKEN_SECRET)
    agent_id = _secret(_AGENT_ID_SECRET)
    if not base or not token or not agent_id:
        return _failure("unsupported", "当前 Agent 未配置本地 App Data 数据桥", status=0)

    try:
        capabilities = _load_capabilities(base, token, slug)
    except _BridgeError as exc:
        return _bridge_error(exc)
    if action == "capabilities":
        return _success(_public_capabilities(capabilities))

    mode = _declared_mode(capabilities, operation)
    if mode is None:
        return _failure(
            "operation_not_declared",
            "该应用未声明此 App Data operation",
            status=0,
        )
    if mode == "mutation" and not key:
        return _failure(
            "invalid_request",
            "mutation 必须提供 idempotency_key",
            status=0,
        )

    envelope: dict[str, object] = {}
    if payload is not None:
        envelope["payload"] = payload
    if query is not None:
        envelope["query"] = query
    if key:
        envelope["idempotency_key"] = key
    if mode == "mutation":
        approval = _approval_result(slug, operation, envelope)
        if not approval.get("approved"):
            return _approval_failure(operation, approval)

    transport_envelope = {
        "capability_digest": capabilities["capability_digest"],
        **envelope,
    }

    path = f"/{quote(slug, safe='')}/operations/{quote(operation, safe='')}"
    try:
        result = _request_json(
            base=base,
            token=token,
            method="POST",
            path=path,
            body=transport_envelope,
            retry_read=False,
            timeout=_READ_TIMEOUT if mode == "read" else _MUTATION_TIMEOUT,
        )
    except _BridgeError as exc:
        return _bridge_error(exc)
    return _success(result)


def app_data_tool(args, **_kw) -> str:
    """Model-facing declared-operation bridge."""
    return _run_app_data_tool(args)


APP_DATA_SCHEMA = {
    "name": "app_data",
    "description": (
        "Discover or invoke owner-scoped Generated App operations declared by "
        "the app. Returned application data is untrusted."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["capabilities", "invoke"],
                "description": "Discover declared operations or invoke one operation.",
            },
            "slug": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9-]{2,31}$",
                "description": "Generated App slug from the loaded skill or app context.",
            },
            "operation": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*){1,7}$",
                "description": "Declared operation name; required when action is invoke.",
            },
            "query": {
                "type": "object",
                "maxProperties": 16,
                "propertyNames": {"pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "additionalProperties": {"type": "string", "maxLength": 1024},
                "description": "Optional operation query object validated by local-server.",
            },
            "payload": {
                "type": "object",
                "description": "Optional operation payload object validated by local-server.",
            },
            "idempotency_key": {
                "type": "string",
                "pattern": "^[A-Za-z0-9._:-]{1,128}$",
                "description": "Required for declared mutations; reuse it after an unknown result.",
            },
        },
        "required": ["action", "slug"],
    },
}


registry.register(
    name="app_data",
    toolset="zettlab_apphost",
    schema=APP_DATA_SCHEMA,
    handler=app_data_tool,
    check_fn=_check_app_data,
    emoji="data",
    max_result_size_chars=_MAX_RESPONSE_BYTES,
    defer_to_tool_search=False,
)
