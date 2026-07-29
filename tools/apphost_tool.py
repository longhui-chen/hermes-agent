"""App Host Tool — hermes-mediated access to local-server's App Host internal face.

Device-hosted generated applications are managed through local-server's App Host
internal endpoints, which require the per-agent action token. That token also
opens the connector bridge and ai-proxy faces, so it must never reach
model-written shell. The split is: shell does only filesystem work (copy
template, write sources, build, self-test), while every credentialed call goes
through this tool — hermes resolves the token and base URL from the active
profile's secret scope (``agent.secret_scope.get_secret``), which works both in
the shared multiplexing gateway (values live in the profile ``.env``, NOT in
``os.environ``) and in the single-profile process (values live in the env).

The install/reload request bodies carry only path strings; the filesystem is
the hand-off medium between shell and API.
"""

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from agent.secret_scope import get_secret

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_DEFAULT_TIMEOUT = 30.0
# install/reload need headroom over the server's own pipeline (Start alone is
# capped at 30s, selfCheck adds 5s), and the stakes are asymmetric: the server
# derives its work context from the REQUEST context, so a client-side timeout
# cancels the request and triggers rollbackInstall — deleting an app that was
# about to install fine. Never auto-retry these here; retry semantics belong
# to the calling skill (reload is idempotent per commit, this layer is not the
# place to judge).
_LONG_TIMEOUT = 120.0
# acquire_slot may queue behind other builds; the model may ask for a longer
# bounded wait, but never an unbounded one.
_MAX_WAIT_SECONDS = 300
_MAX_RESPONSE_BYTES = 256 * 1024
_DEFAULT_LOG_TAIL = 200
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_LIFECYCLE_ACTIONS = ("start", "stop", "restart")
_HTTP_ACTIONS = (
    "probe", "list", "acquire_slot", "release_slot", "install", "reload",
    "delete", "recover", "lifecycle", "logs",
)
_ACTIONS = _HTTP_ACTIONS + ("build_env",)

APP_HOST_SCHEMA = {
    "name": "app_host",
    "description": (
        "Manage device-hosted generated applications via the local App Host. "
        "Hermes holds the credentials and performs the HTTP calls — never try "
        "to reach App Host endpoints from shell. Actions: probe (capability + "
        "storage headroom check), list (installed apps), acquire_slot / "
        "release_slot (build-slot admission before compiling; acquire returns "
        "a slot token, or queue_ahead while queued), install (register an app "
        "staged on disk), reload (rebuild + restart from a staging dir; "
        "idempotent — resending the same commit returns current state), delete "
        "(soft-delete into the recycle bin), recover, lifecycle "
        "(start/stop/restart), logs (recent log tail), build_env (local check "
        "of the shared Go vendor dir to copy into the staging area; makes no "
        "HTTP request)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "The App Host operation to perform.",
            },
            "slug": {
                "type": "string",
                "description": (
                    "Application slug. Required for install, reload, delete, "
                    "recover, lifecycle, and logs."
                ),
            },
            "staging_dir": {
                "type": "string",
                "description": (
                    "Absolute path of the staged application source on the "
                    "device. Required for install and reload."
                ),
            },
            "lifecycle_action": {
                "type": "string",
                "enum": list(_LIFECYCLE_ACTIONS),
                "description": "Required for the lifecycle action.",
            },
            "tail": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "default": _DEFAULT_LOG_TAIL,
                "description": "For logs: number of trailing lines to return.",
            },
            "slot_token": {
                "type": "string",
                "description": "Required for release_slot: the token returned by acquire_slot.",
            },
            "wait_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_WAIT_SECONDS,
                "description": (
                    "For acquire_slot only: bounded wait for a build slot "
                    f"(default {int(_DEFAULT_TIMEOUT)}, max {_MAX_WAIT_SECONDS})."
                ),
            },
        },
        "required": ["action"],
    },
}


def _secret(name):
    return str(get_secret(name, "") or "").strip()


def _base_url():
    """Validated App Host base URL from the profile secret scope, or None."""
    raw = _secret("ZET_APPHOST_BASE_URL")
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return raw.rstrip("/")


def _check_app_host():
    """Expose the tool only when both the App Host base URL and the agent
    action token are resolvable in the active profile scope."""
    return bool(_base_url() and _secret("ZETTLAB_AGENT_ACTION_TOKEN"))


# Availability depends on the per-turn profile scope; the registry must not
# serve one profile's cached verdict to another (see registry._must_recheck_
# profile_scope).
_check_app_host._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _ok(data):
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _fail(error, status=None):
    """Failure envelope. ``error`` is the upstream {code, message} error body
    verbatim — skills branch on the ``code`` string (never the HTTP status),
    so it must never be flattened into prose. Locally-produced errors mimic
    the same shape."""
    return json.dumps({"ok": False, "error": error, "status": status}, ensure_ascii=False)


def _local_error(code, message, status=None):
    """A locally-produced failure in the upstream error-body shape, so skills
    need only one set of branches. The message must never contain the token
    or the full base URL."""
    return _fail({"code": code, "message": message}, status=status)


class _BadRequest(ValueError):
    """Model-facing validation error (message is safe to return verbatim)."""


def _require_slug(args):
    slug = str(args.get("slug", "") or "").strip()
    if not slug:
        raise _BadRequest("该动作需要提供 slug 参数")
    if not _SLUG_RE.match(slug):
        raise _BadRequest("slug 格式不合法（仅允许字母、数字、点、下划线、连字符）")
    return slug


def _require_staging_dir(args):
    staging_dir = str(args.get("staging_dir", "") or "").strip()
    if not staging_dir:
        raise _BadRequest("该动作需要提供 staging_dir 参数")
    return staging_dir


def _build_request(action, args):
    """Return (method, path, body_dict_or_None, timeout) for an HTTP action."""
    timeout = _DEFAULT_TIMEOUT
    if action == "probe":
        return "GET", "/storage", None, timeout
    if action == "list":
        return "GET", "", None, timeout
    if action == "acquire_slot":
        wait = args.get("wait_seconds")
        if wait is not None:
            try:
                timeout = float(min(max(int(wait), 1), _MAX_WAIT_SECONDS))
            except (TypeError, ValueError):
                raise _BadRequest("wait_seconds 必须是整数")
        return "POST", "/buildslot", None, timeout
    if action == "release_slot":
        slot_token = str(args.get("slot_token", "") or "").strip()
        if not slot_token:
            raise _BadRequest("release_slot 需要提供 slot_token 参数")
        return "DELETE", "/buildslot/" + quote(slot_token, safe=""), None, timeout
    if action == "install":
        return "POST", "/install", {
            "staging_dir": _require_staging_dir(args),
            "slug": _require_slug(args),
        }, _LONG_TIMEOUT
    if action == "reload":
        slug = _require_slug(args)
        return "POST", f"/{slug}/reload", {"staging_dir": _require_staging_dir(args)}, _LONG_TIMEOUT
    if action == "delete":
        return "DELETE", f"/{_require_slug(args)}", None, timeout
    if action == "recover":
        return "POST", f"/{_require_slug(args)}/recover", None, timeout
    if action == "lifecycle":
        slug = _require_slug(args)
        lifecycle_action = str(args.get("lifecycle_action", "") or "").strip()
        if lifecycle_action not in _LIFECYCLE_ACTIONS:
            raise _BadRequest("lifecycle 需要 lifecycle_action 参数（start/stop/restart）")
        return "POST", f"/{slug}/lifecycle", {"action": lifecycle_action}, timeout
    if action == "logs":
        slug = _require_slug(args)
        tail = args.get("tail", _DEFAULT_LOG_TAIL)
        try:
            tail = min(max(int(tail), 1), 10000)
        except (TypeError, ValueError):
            raise _BadRequest("tail 必须是整数")
        return "GET", f"/{slug}/logs?tail={tail}", None, timeout
    raise _BadRequest(f"未知动作：{action}")


def _parse_upstream_error(raw_body):
    """The upstream JSON error body ({code, message}) verbatim, or None when
    the body is absent / not a JSON object (degrade to transport_error)."""
    if not raw_body or len(raw_body) > _MAX_RESPONSE_BYTES:
        return None
    try:
        parsed = json.loads(raw_body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_env_result():
    vendor_dir = _secret("ZETTLAB_GO_VENDOR_DIR")
    ready = bool(vendor_dir) and os.path.isdir(vendor_dir)
    return _ok({"vendor_dir": vendor_dir, "ready": ready})


def app_host_tool(args, **_kw):
    # Tool handlers must return a STRING (json-encoded) — a raw dict reaches
    # the model provider as non-string content and gets rejected (same
    # contract as list_my_channels).
    args = args or {}
    action = str(args.get("action", "") or "").strip()

    if action == "build_env":
        # Pure local check — no HTTP request, no credentials leave the tool.
        return _build_env_result()

    try:
        method, path, body, timeout = _build_request(action, args)
    except _BadRequest as exc:
        return _local_error("invalid_request", str(exc))

    base = _base_url()
    token = _secret("ZETTLAB_AGENT_ACTION_TOKEN")
    if not base or not token:
        return _local_error("unsupported", "App Host 未配置或不可用，这台设备暂不支持生成应用")

    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {_ACTION_TOKEN_HEADER: token, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        upstream = _parse_upstream_error(exc.read() or b"")
        if upstream is not None:
            # Verbatim pass-through: skills branch on the upstream `code`
            # string (slug_conflict / storage_full / ...), never the HTTP
            # status. Do not flatten into prose.
            return _fail(upstream, status=exc.code)
        return _local_error(
            "transport_error",
            f"App Host 请求失败（HTTP {exc.code}），未返回可解析的错误体",
            status=exc.code,
        )
    except Exception:
        # Never echo the exception: URLError/timeout messages can embed the
        # request URL.
        return _local_error("transport_error", "无法连接 App Host 服务")

    if len(raw) > _MAX_RESPONSE_BYTES:
        return _local_error("transport_error", "App Host 返回内容过大", status=status)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return _local_error("transport_error", "App Host 返回了无法解析的内容", status=status)
    return _ok(parsed)


from tools.registry import registry  # noqa: E402

registry.register(
    name="app_host",
    toolset="zettlab_apphost",
    schema=APP_HOST_SCHEMA,
    handler=app_host_tool,
    check_fn=_check_app_host,
    emoji="🏗️",
)
