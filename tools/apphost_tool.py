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

The preferred publish request carries only an agent-output-relative path; the
filesystem is the hand-off medium between shell and API. Legacy install/reload
remain available for devices and skills that still stage through App Host.
"""

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from agent.credential_broker import request_app_auto_refresh_token
from agent.secret_scope import get_secret

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_AGENT_ID_SECRET = "ZET_AGENT_ID"
_DEFAULT_TIMEOUT = 30.0
# install/reload need headroom over the server's own pipeline (Start alone is
# capped at 30s, selfCheck adds 5s), and the stakes are asymmetric: the server
# derives its work context from the REQUEST context, so a client-side timeout
# cancels the request and triggers rollbackInstall — deleting an app that was
# about to install fine. Never auto-retry these here; retry semantics belong
# to the calling skill (reload is idempotent per commit, this layer is not the
# place to judge). acquire_slot shares this tier: the server answers a queued
# caller immediately (queue_ahead), but a GRANTED slot pays the shared
# build-environment integrity walk (~25k files) before responding — slow on a
# busy 1-core board, and a client-side cut here leaks the slot until its TTL.
_LONG_TIMEOUT = 120.0
# Must exceed the server's single-logs-response cap (512 KiB,
# apphost/service.go); a smaller cap makes every long-log fetch fail as
# status=200 + transport_error, which the skill reads as a transient outage
# and retries forever.
_MAX_RESPONSE_BYTES = 1024 * 1024
# Delivery-layer cap for text payloads (logs): the transport cap above keeps
# the fetch from failing, but half a megabyte of log text poured into the
# model's context is its own harm — the model needs the tail (errors are
# usually at the end), not the whole window. 64 KiB comfortably holds the
# default tail=200 lines while staying a small fraction of any context.
_MAX_TEXT_PAYLOAD_CHARS = 64 * 1024
_DEFAULT_LOG_TAIL = 200
_MAX_STAGING_DIR_CHARS = 1024
_MAX_SOURCE_SUBDIR_CHARS = 1024
_MAX_APP_PATH_CHARS = 1024
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Credentialed loopback transport (no env proxies, no redirects) — shared
# with the other action-token call sites via tools.loopback_transport; the
# local alias keeps this module's patch point stable.
from tools.loopback_transport import (  # noqa: E402
    HARDENED_OPENER as _NO_PROXY_OPENER,
    is_trusted_loopback_http as _is_trusted_loopback_http,
)


def _urlopen(req, timeout):
    return _NO_PROXY_OPENER.open(req, timeout=timeout)

_LIFECYCLE_ACTIONS = ("start", "stop", "restart")
_PUBLISH_MODES = ("install", "reload")
# Installing an app forces an answer to "does this app's data need to refresh
# on its own?". Five real device runs showed the question being skipped in
# silence — the skill text asks for it, and the run still walks past it and
# ships a dashboard with a manual button after the user asked for a daily
# fetch. A skill can be out-argued by another skill; a required tool argument
# cannot. "user_confirmed_auto" is also recorded server-side, so an app the
# user asked to self-refresh that never got a maintainer is a fact someone can
# query later instead of a promise that quietly evaporated.
_DATA_REFRESH_CHOICES = (
    "static", "external_unconfirmed", "user_confirmed_auto", "user_declined"
)
# The hidden maintainer gets one write capability, not the app's whole HTTP
# surface. Generated apps expose POST /api/refresh as the user-confirmed data
# maintenance verb; every other write path stays unavailable to model calls.
# Reads use declared typed AppOperation capabilities. This legacy call action
# exposes only the user-confirmed refresh mutation.
_CALL_HTTP_METHODS = ("POST",)
_CALL_WRITE_PATHS = frozenset({"/api/refresh"})
# NOTE: no "recover" — the internal (agent) face deliberately does not expose
# it (an action token authenticates one agent, not the device); recovery from
# the recycle bin lives on the JWT member face, i.e. the client app's list.
_HTTP_ACTIONS = (
    "probe", "list", "acquire_slot", "release_slot", "publish", "install",
    "reload", "rollback", "delete", "lifecycle", "logs", "call",
    # Typed AppOperation endpoints are intentionally distinct from the
    # operation journal: the former invokes a declared app capability, the
    # latter only reads the state of an already accepted workflow.
    "app_capabilities", "app_operation", "workflow_operation_status", "workflow_operation_resume",
)
_ACTIONS = _HTTP_ACTIONS + ("build_env",)

APP_HOST_SCHEMA = {
    "name": "app_host",
    "description": (
        "Manage device-hosted generated applications via the local App Host. "
        "Hermes holds the credentials and performs the HTTP calls — never try "
        "to reach App Host endpoints from shell. Actions: probe (capability + "
        "storage headroom check), list (installed apps), acquire_slot / "
        "release_slot (build-slot admission before compiling; acquire answers "
        "immediately with a slot token, or queue_ahead while queued — poll by "
        "calling again), publish (formal install/update from the current "
        "agent's output workspace: securely copy a generated app, then install "
        "or reload it; the app name comes from metadata.json; on devices whose "
        "local-server predates this route it fails with code \"unsupported\" "
        "AND HTTP status 404 or 409 — only that pair means the route is "
        "missing, and only then does the calling skill fall back to "
        "install/reload; an \"unsupported\" carrying any other status is a "
        "different problem and falling back cannot help it), install "
        "(register an app from a directory staged directly under App Host's "
        ".staging root; the fallback creation path on devices without publish "
        "support), reload "
        "(rebuild + restart from a staging dir — the fallback update path when "
        "publish is unsupported; idempotent — resending the "
        "same commit returns current state), rollback (put the previous "
        "version back — one step, no rebuild; requires to_version from the "
        "app's prev_version_id so a retry cannot swap it forward again), "
        "delete (soft-delete into the "
        "recycle bin; recovery is done from the client app's list, there is "
        "no recover action here), lifecycle (start/stop/restart), logs "
        "(recent log tail), call (invoke the single bounded HTTP capability of an "
        "app bound to the current hidden maintainer: POST /api/refresh from the user-confirmed "
        "automatic-refresh flow. Give the app's slug and API path, with http_method "
        "and an optional JSON body; never a full URL, host or port — the "
        "host resolves the target from the slug, and only the owning agent "
        "can reach the app. The result's data.status / data.body are the "
        "APP's answer: an app-side 4xx/5xx there still means the call went "
        "through — read the app's error and fix the request, do not treat it "
        "as a tool failure or blindly retry. Only a failed forwarding chain "
        "returns ok:false, and its error code decides retries — the error "
        "body also carries a retryable flag: app_updating / app_waking are "
        "transient, wait ~5s and retry; app_stopped means the user stopped "
        "the app on purpose — never retry and never try to start or restart "
        "it; app_broken / app_start_failed / app_unreachable / not_found do "
        "not heal by retrying — check logs or report instead; code "
        "\"unsupported\" with HTTP status 404 means this device's "
        "local-server predates the call route: the device cannot forward "
        "calls at all, which is NOT the same as the app missing — stop, do "
        "not keep probing other paths or slugs), build_env (local check of "
        "the shared Go vendor "
        "dir to copy into the build workspace; makes no HTTP request)."
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
                    "Application slug. Required for install, reload, "
                    "rollback, delete, lifecycle, logs, and call."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Required for call: the app's own API path, starting "
                    "with '/'. It must be exactly /api/refresh, without a "
                    "query string. General reads and writes require declared "
                    "typed app capabilities. Only the path within the app — never a full URL, "
                    "host or port; the host resolves the target from the "
                    "slug."
                ),
            },
            "http_method": {
                "type": "string",
                "enum": list(_CALL_HTTP_METHODS),
                "description": (
                    "Required for call: the HTTP method of the app request."
                ),
            },
            "body": {
                "type": "object",
                "description": (
                    "For call: JSON request body forwarded to the app as-is. "
                    "Meaningful only with a non-GET http_method."
                ),
            },
            "mode": {
                "type": "string",
                "enum": list(_PUBLISH_MODES),
                "description": (
                    "Required for publish: install creates a new app; reload "
                    "updates an existing app owned by the current agent."
                ),
            },
            "source_subdir": {
                "type": "string",
                "description": (
                    "Legacy publish override only. Leave omitted for a new "
                    "application: App Host publishes the sole verified build "
                    "from the current session automatically."
                ),
            },
            "data_refresh": {
                "type": "string",
                "enum": list(_DATA_REFRESH_CHOICES),
                "description": (
                    "Required for publish(mode=install), and recorded by "
                    "legacy action=install for non-automatic apps. Does this app's data "
                    "need to keep refreshing on its own? "
                    "static = the user types the data in themselves (ledger, "
                    "to-do, notes) and nothing outside the device changes it. "
                    "external_unconfirmed = the data comes from outside, but "
                    "the device could not ask for refresh consent because the "
                    "optional capability was unavailable. "
                    "user_confirmed_auto = the data comes from outside and the "
                    "user agreed to a schedule — it requires publish(mode=install) "
                    "with a complete operation; legacy install is forbidden. "
                    "user_declined = you asked and the user said no. "
                    "Answer from what the user actually said, not from what the "
                    "app could get away with: an app that shows prices, weather "
                    "or rates and only has a manual refresh button is not static."
                ),
            },
            "staging_dir": {
                "type": "string",
                "description": (
                    "Absolute path of a direct child of App Host's .staging "
                    "root. Required for install and reload (the fallback path "
                    "when publish is unsupported on this device)."
                ),
            },
            "note": {
                "type": "string",
                "description": (
                    "For publish(mode=reload) or reload: one line "
                    "saying what this change did, in the "
                    "user's own words (\u201cFooter \u52a0\u4e86\u4e00\u4e2a\u94fe\u63a5\u201d). It is stored with the "
                    "version and is what the user is shown when deciding "
                    "whether to undo it — without it an undo can only offer a "
                    "nameless version."
                ),
            },
            "to_version": {
                "type": "string",
                "description": (
                    "Required for rollback: the version to go back to, taken "
                    "from the app's prev_version_id in list. Naming the "
                    "version is what makes the request safe to retry — an "
                    "undo that already succeeded is recognised instead of "
                    "being applied a second time and swapping the app "
                    "forward again."
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
            "app_operation": {
                "type": "string",
                "description": "Required for app_operation: declared AppOperation name.",
            },
            "payload": {
                "type": "object",
                "description": "Required for app_operation: operation payload, passed unchanged inside App Host's closed envelope.",
            },
            "query": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional app_operation query values declared by the app capability.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Optional app_operation idempotency key declared by the app capability.",
            },
            "capability_digest": {
                "type": "string",
                "description": "Required for app_operation: digest returned by app_capabilities for the declared operation contract.",
            },
            "operation": {
                "type": "object",
                "description": "Optional immutable workflow operation intent for publish(mode=install/reload), passed unchanged to App Host as operation. Legacy install and reload do not support workflow operations.",
            },
            "operation_id": {
                "type": "string",
                "description": "Required for workflow_operation_status: App Host operation journal receipt id.",
            },
        },
        "required": ["action"],
    },
}


def _secret(name):
    return str(get_secret(name, "") or "").strip()


def _base_url():
    """Validated App Host base URL from the profile secret scope, or None.

    The internal face is plain HTTP on loopback only; anything else means the
    value was repointed somewhere the action token must not go — fail closed
    rather than hand the credential to whoever answers."""
    raw = _secret("ZET_APPHOST_BASE_URL")
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if not parts.netloc or not _is_trusted_loopback_http(parts):
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


# The `status` field is a three-way contract for the caller's retry decision:
# an HTTP code means the server answered; null means the request WENT OUT but
# the outcome is unknown (timeout / dropped connection — may have succeeded,
# retry per idempotent semantics); 0 means the tool rejected the call locally
# and not a single byte was sent — retrying verbatim is pointless, the call
# must be corrected first.
#
# 0 is falsy: any consumer distinguishing the tiers must compare strictly
# (`is None` / `== 0`), never truthiness — `if not status` conflates the two
# tiers whose handling is opposite. This module itself never branches on the
# envelope's status (it is output-only here); keep it that way.
_STATUS_NOT_SENT = 0


def _fail(error, status=None):
    """Failure envelope. ``error`` is the upstream {code, message} error body
    verbatim — skills branch on the ``code`` string (never the HTTP status),
    so it must never be flattened into prose. Locally-produced errors mimic
    the same shape."""
    return json.dumps({"ok": False, "error": error, "status": status}, ensure_ascii=False)


def _local_error(code, message, *, status):
    """A locally-produced failure in the upstream error-body shape, so skills
    need only one set of branches. The message must never contain the token
    or the full base URL.

    ``status`` is deliberately REQUIRED, with no default: a default of None
    would make a forgotten argument silently report a local rejection as
    "request went out, outcome unknown" — the tier whose handling (idempotent
    resend) is exactly wrong for a call that never left the tool. Forgetting
    must be a TypeError at the call site, not a wrong retry downstream. Pass
    ``_STATUS_NOT_SENT`` when nothing was sent, ``None`` only for a request
    that went out without an answer, or the HTTP code when the server spoke.
    """
    return _fail({"code": code, "message": message}, status=status)


class _BadRequest(ValueError):
    """Model-facing validation error (message is safe to return verbatim)."""


class _AutoRefreshScopeUnavailable(ValueError):
    """The automatic-maintenance capability was not minted or is unsafe to use."""


def _require_slug(args):
    slug = str(args.get("slug", "") or "").strip()
    if not slug:
        raise _BadRequest("该动作需要提供 slug 参数")
    if not _SLUG_RE.match(slug):
        raise _BadRequest("slug 格式不合法（仅允许字母、数字、点、下划线、连字符）")
    return slug


def _require_staging_dir(args):
    """String-level precheck of the model-supplied staging path.

    The server is the authoritative gate (validateStagingPath: direct child
    of its staging root, Lstat symlink refusal — checks only it can make with
    the real filesystem view). This layer rejects the obviously-malformed
    forms locally so they never ride a credentialed request: relative paths,
    parent-directory traversal, embedded NUL/newlines, absurd length.
    """
    staging_dir = str(args.get("staging_dir", "") or "").strip()
    if not staging_dir:
        raise _BadRequest("该动作需要提供 staging_dir 参数")
    if len(staging_dir) > _MAX_STAGING_DIR_CHARS:
        raise _BadRequest("staging_dir 过长")
    if any(ch in staging_dir for ch in ("\x00", "\n", "\r")):
        raise _BadRequest("staging_dir 含非法字符")
    if not staging_dir.startswith("/"):
        raise _BadRequest("staging_dir 必须是绝对路径")
    if ".." in staging_dir.split("/"):
        raise _BadRequest("staging_dir 不允许包含上级目录段")
    return staging_dir


def _require_source_subdir(args):
    """Validate the portable path handed to local-server's output resolver.

    The server performs the authoritative descriptor-confined walk. This
    check keeps absolute paths, traversal and platform-specific separators
    from ever riding a credentialed request.
    """
    source_subdir = str(args.get("source_subdir", "") or "").strip()
    if not source_subdir:
        return ""
    if len(source_subdir) > _MAX_SOURCE_SUBDIR_CHARS:
        raise _BadRequest("source_subdir 过长")
    if any(ch in source_subdir for ch in ("\x00", "\n", "\r", "\\")):
        raise _BadRequest("source_subdir 含非法字符")
    if source_subdir.startswith("/"):
        raise _BadRequest("source_subdir 必须是当前 Agent output 下的相对路径")
    parts = source_subdir.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise _BadRequest("source_subdir 不允许空目录段、当前目录段或上级目录段")
    return source_subdir


def _require_app_path(args):
    """String-level precheck of the model-supplied in-app path for call.

    The server is the authoritative gate (it builds the target URL from the
    slug itself and validates the path again). This layer rejects the
    obviously-malformed forms locally so they never ride a credentialed
    request: full URLs, host-relative ``//`` forms, traversal segments,
    control characters, absurd length.
    """
    path = str(args.get("path", "") or "").strip()
    if not path:
        raise _BadRequest("call 需要提供 path 参数（应用自身的 API 路径，如 /api/refresh）")
    if len(path) > _MAX_APP_PATH_CHARS:
        raise _BadRequest("path 过长")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        raise _BadRequest("path 含非法字符")
    if "://" in path:
        raise _BadRequest("path 必须是应用内的相对路径，不能是完整 URL")
    if not path.startswith("/"):
        raise _BadRequest("path 必须以 / 开头")
    if path.startswith("//"):
        raise _BadRequest("path 不能以 // 开头")
    if ".." in path.split("/"):
        raise _BadRequest("path 不允许包含上级目录段")
    return path


def _require_http_method(args):
    method = str(args.get("http_method", "") or "").strip().upper()
    if method not in _CALL_HTTP_METHODS:
        raise _BadRequest(
            "call 需要 http_method 参数（POST）"
        )
    return method


def _session_key():
    """Stable chat-session identity for creation provenance, or "".

    Only HERMES_SESSION_KEY qualifies: the lineage id in HERMES_SESSION_ID is
    rotated by context compaction, so recording it as provenance would name a
    session that stops being findable mid-conversation. No session key means
    no provenance — the server treats the field as optional.
    """
    try:
        from gateway.session_context import get_session_env

        value = get_session_env("HERMES_SESSION_KEY", "")
    except Exception:
        value = ""
    if not value:
        # Outside a bound request the ContextVar getter deliberately returns
        # its empty default; CLI tests and the single-profile daemon still
        # use the legacy process environment in that case.
        value = os.environ.get("HERMES_SESSION_KEY", "")
    return str(value or "").strip()


def _execution_headers():
    """Forward server-issued execution context; model arguments never shape it."""
    try:
        from gateway.session_context import (
            business_execution_action,
            business_execution_action_version,
            current_turn_identity,
            get_session_env,
        )
        token = str(business_execution_action() or "").strip()
        version = str(business_execution_action_version() or "").strip()
        identity = current_turn_identity()
        turn_id = identity[0] if identity else ""
        session_id = str(get_session_env("HERMES_SESSION_ID", "") or "").strip()
    except Exception:
        return {}
    headers = {}
    if token:
        headers["X-Zettlab-Business-Execution-Action"] = token
        headers["X-Zettlab-Business-Execution-Action-Version"] = version
    if turn_id:
        headers["X-Hermes-Turn-Id"] = str(turn_id)
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id
    # Bound scheduler sessions are server-generated as
    # cron_task_<job-id>_<UTC timestamp>.
    # The task id is therefore derived from trusted execution context, never
    # supplied by a model tool argument.
    match = re.fullmatch(r"cron_task_([a-f0-9]{12})_\d{8}_\d{6}", session_id)
    if match:
        headers["X-Zettlab-App-Maintenance-Task-Id"] = match.group(1)
    return headers


def _auto_refresh_scope_token(action, body, execution_headers):
    """Mint the one-shot scope for a user-confirmed App Host operation.

    ``user_confirmed_auto`` is durable user intent in the immutable operation;
    the execution headers bind this particular publication to the active user
    turn.  The model never receives the resulting bearer: it is sent once to
    App Host, which claims it against the operation fingerprint before it can
    provision the maintainer and cron job.
    """
    if action != "publish" or not isinstance(body, dict):
        return None
    operation = body.get("operation")
    if not isinstance(operation, dict) or operation.get("data_refresh") != "user_confirmed_auto":
        return None

    required_execution_headers = {
        "X-Zettlab-Business-Execution-Action",
        "X-Zettlab-Business-Execution-Action-Version",
        "X-Hermes-Turn-Id",
        "X-Hermes-Session-Id",
    }
    if not required_execution_headers.issubset(execution_headers):
        raise _AutoRefreshScopeUnavailable(
            "自动维护只能在当前已验证的用户会话中发布；未发送发布请求"
        )
    agent_id = _secret(_AGENT_ID_SECRET)
    if not agent_id:
        raise _AutoRefreshScopeUnavailable(
            "当前 Agent 身份不可用，无法授权自动维护；未发送发布请求"
        )
    try:
        token = request_app_auto_refresh_token(agent_id)
    except Exception:
        raise _AutoRefreshScopeUnavailable(
            "自动维护授权暂不可用；未发送发布请求"
        ) from None
    if re.fullmatch(r"[0-9a-f]{64}", token or "") is None:
        raise _AutoRefreshScopeUnavailable(
            "自动维护授权无效；未发送发布请求"
        )
    return token


def _build_request(action, args):
    """Return (method, path, body_dict_or_None, timeout) for an HTTP action."""
    timeout = _DEFAULT_TIMEOUT
    if action == "probe":
        return "GET", "/storage", None, timeout
    if action == "list":
        # Owner-scoped on purpose: the unfiltered device-wide list shows apps
        # the owner gates on reload/delete/lifecycle/logs would then 404 —
        # the model must only see what it can act on.
        return "GET", "?mine=1", None, timeout
    if action == "app_capabilities":
        return "GET", f"/{_require_slug(args)}/capabilities", None, timeout
    if action == "app_operation":
        slug = _require_slug(args)
        operation = str(args.get("app_operation", "") or "").strip()
        if not _SLUG_RE.match(operation):
            raise _BadRequest("app_operation 需要合法的 operation 名称")
        payload = args.get("payload")
        if not isinstance(payload, dict):
            raise _BadRequest("app_operation 需要 object 类型的 payload")
        capability_digest = str(args.get("capability_digest", "") or "").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", capability_digest):
            raise _BadRequest("app_operation 需要 64 位 capability_digest")
        body = {"payload": payload, "capability_digest": capability_digest.lower()}
        query = args.get("query")
        if query is not None:
            if not isinstance(query, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in query.items()):
                raise _BadRequest("app_operation 的 query 必须是 string map")
            body["query"] = query
        idempotency_key = str(args.get("idempotency_key", "") or "").strip()
        if idempotency_key:
            body["idempotency_key"] = idempotency_key
        return "POST", f"/{slug}/operations/{quote(operation, safe='')}", body, timeout
    if action == "workflow_operation_status":
        slug = _require_slug(args)
        operation_id = str(args.get("operation_id", "") or "").strip()
        if not operation_id or len(operation_id) > 256 or any(ord(ch) < 0x20 for ch in operation_id):
            raise _BadRequest("workflow_operation_status 需要合法的 operation_id")
        return "GET", f"/{slug}/operation/{quote(operation_id, safe='')}", None, timeout
    if action == "workflow_operation_resume":
        slug = _require_slug(args)
        operation_id = str(args.get("operation_id", "") or "").strip()
        if not operation_id or len(operation_id) > 256 or any(ord(ch) < 0x20 for ch in operation_id):
            raise _BadRequest("workflow_operation_resume 需要合法的 operation_id")
        return "POST", f"/{slug}/operation/{quote(operation_id, safe='')}/resume", None, _LONG_TIMEOUT
    if action == "acquire_slot":
        # Non-blocking on the server (queued → immediate queue_ahead; poll by
        # calling again), but a GRANTED slot pays the integrity walk before
        # the response — long tier, see _LONG_TIMEOUT.
        return "POST", "/buildslot", None, _LONG_TIMEOUT
    if action == "release_slot":
        slot_token = str(args.get("slot_token", "") or "").strip()
        if not slot_token:
            raise _BadRequest("release_slot 需要提供 slot_token 参数")
        return "DELETE", "/buildslot/" + quote(slot_token, safe=""), None, timeout
    if action == "publish":
        mode = str(args.get("mode", "") or "").strip()
        if mode not in _PUBLISH_MODES:
            raise _BadRequest("publish 需要 mode 参数（install/reload）")
        body = {"mode": mode}
        source_subdir = _require_source_subdir(args)
        if source_subdir:
            body["source_subdir"] = source_subdir
        note = str(args.get("note", "") or "").strip()
        if note:
            body["note"] = note
        # Creation provenance rides only on install: a reload updates code,
        # it never rewrites who created the app.
        if mode == "install":
            data_refresh = str(args.get("data_refresh", "") or "").strip()
            if data_refresh not in _DATA_REFRESH_CHOICES:
                raise _BadRequest(
                    "publish(mode=install) 需要 data_refresh 参数（"
                    + "/".join(_DATA_REFRESH_CHOICES)
                    + "）：这个应用的数据要不要自己持续更新？照用户说过的话答"
                )
            body["data_refresh"] = data_refresh
            session_key = _session_key()
            if session_key:
                body["session_id"] = session_key
        operation = args.get("operation")
        if operation is not None:
            if not isinstance(operation, dict):
                raise _BadRequest("operation 必须是 object")
            operation_data_refresh = str(operation.get("data_refresh", "") or "").strip()
            if operation_data_refresh not in _DATA_REFRESH_CHOICES:
                raise _BadRequest("operation 需要有效 data_refresh")
            outer_data_refresh = str(body.get("data_refresh", "") or "").strip()
            if outer_data_refresh and outer_data_refresh != operation_data_refresh:
                raise _BadRequest("operation.data_refresh 必须与 data_refresh 一致")
            # Local Server validates this exact outer/inner equality for both
            # install and reload publication. The immutable intent is the
            # source of truth, so callers never need to duplicate it for reload.
            body["data_refresh"] = operation_data_refresh
            body["operation"] = operation
        elif body.get("data_refresh") == "user_confirmed_auto":
            raise _BadRequest(
                "自动维护必须通过 publish 提供完整 operation，才能原子创建维护者和定时任务"
            )
        return "POST", "/publish", body, _LONG_TIMEOUT
    if action == "install":
        if args.get("operation") is not None:
            # Legacy install predates the journal-aware publication endpoint.
            # Never send or silently drop an immutable operation intent: use
            # publish(mode=install) so App Host can drive the transaction.
            raise _BadRequest("legacy install 不支持 operation；请使用 publish(mode=install)")
        data_refresh = str(args.get("data_refresh", "") or "").strip()
        if data_refresh not in _DATA_REFRESH_CHOICES:
            raise _BadRequest(
                "install 需要 data_refresh 参数（"
                + "/".join(_DATA_REFRESH_CHOICES)
                + "）：这个应用的数据要不要自己持续更新？照用户说过的话答"
            )
        if data_refresh == "user_confirmed_auto":
            raise _BadRequest(
                "legacy install 不能启用 user_confirmed_auto；请使用 "
                "publish(mode=install) 并提供完整 operation，才能原子创建维护者和定时任务"
            )
        body = {
            "staging_dir": _require_staging_dir(args),
            "slug": _require_slug(args),
            "data_refresh": data_refresh,
        }
        session_key = _session_key()
        if session_key:
            body["session_id"] = session_key
        return "POST", "/install", body, _LONG_TIMEOUT
    if action == "reload":
        if args.get("operation") is not None:
            # This legacy route predates the journal-aware publication
            # endpoint. Silently dropping the intent would make the caller
            # believe it received the durable workflow semantics it did not.
            raise _BadRequest("legacy reload 不支持 operation；请使用 publish(mode=reload)")
        slug = _require_slug(args)
        body = {"staging_dir": _require_staging_dir(args)}
        note = str(args.get("note", "") or "").strip()
        if note:
            body["note"] = note
        return "POST", f"/{slug}/reload", body, _LONG_TIMEOUT
    if action == "rollback":
        # No rebuild happens here — the previous version is already compiled —
        # but the app is still stopped, swapped and health-checked, so this
        # sits on the long tier with reload rather than the default one.
        slug = _require_slug(args)
        # to_version is REQUIRED on this face even though the server accepts
        # an untargeted rollback. The swap is symmetric, and a long action can
        # end as status=None (request went out, outcome unknown): retrying an
        # untargeted rollback whose first attempt succeeded would swap the app
        # forward again and report success. With the version named, the server
        # recognises an already-done undo (already_done) instead of undoing
        # the undo — so the only retry-safe request shape is the one with a
        # target, and the model always has one (list's prev_version_id).
        to_version = str(args.get("to_version", "") or "").strip()
        if not to_version:
            raise _BadRequest(
                "rollback 需要提供 to_version 参数（取 list 结果中该应用的 prev_version_id）"
            )
        return "POST", f"/{slug}/rollback", {"to_version": to_version}, _LONG_TIMEOUT
    if action == "delete":
        return "DELETE", f"/{_require_slug(args)}", None, timeout
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
    if action == "call":
        # Default tier on purpose: the server's own budget is wake 10s +
        # app response 15s ≈ 25s, deliberately BELOW this 30s — the server
        # must time out first so the failure arrives as a structured error
        # code (app_waking / app_unreachable), not as a client-side
        # status=null transport_error.
        slug = _require_slug(args)
        path = _require_app_path(args)
        method = _require_http_method(args)
        if method != "POST" or path not in _CALL_WRITE_PATHS:
            raise _BadRequest(
                "call 只允许用户已确认自动更新流程使用的精确能力 "
                "POST /api/refresh；读取和其他写入须走应用声明的能力"
            )
        body = {"method": method, "path": path}
        if args.get("body") is not None:
            body["body"] = args["body"]
        return "POST", f"/{slug}/call", body, timeout
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


def _text_payload(text):
    """Non-JSON 2xx payload (logs), truncation-aware.

    Always the same shape — ``text`` (tail-truncated when over the cap),
    ``truncated``, ``total_chars`` — so the caller can tell it saw a partial
    window and, if needed, re-fetch with a smaller ``tail``.
    """
    total = len(text)
    if total <= _MAX_TEXT_PAYLOAD_CHARS:
        return {"text": text, "truncated": False, "total_chars": total}
    return {
        "text": text[-_MAX_TEXT_PAYLOAD_CHARS:],
        "truncated": True,
        "total_chars": total,
    }


def _build_env_result():
    vendor_dir = _secret("ZETTLAB_GO_VENDOR_DIR")
    ready = bool(vendor_dir) and os.path.isdir(vendor_dir)
    return _ok({"vendor_dir": vendor_dir, "ready": ready})


_COMPLETION_STATUS = {
    "probe": {200}, "list": {200}, "app_capabilities": {200},
    "app_operation": {200}, "acquire_slot": {200}, "release_slot": {204},
    "publish": {200, 202}, "install": {200, 202}, "reload": {200, 202}, "rollback": {200},
    "delete": {204}, "lifecycle": {200}, "logs": {200}, "call": {200},
    "workflow_operation_status": {200, 202}, "workflow_operation_resume": {200, 202},
}


def _workflow_operation_status(status, parsed, requested_operation_id):
    if not isinstance(parsed, dict):
        return _local_error("outcome_unknown", "operation journal returned no object", status=status)
    operation_id, terminal = parsed.get("operation_id"), parsed.get("terminal")
    if not isinstance(operation_id, str) or not operation_id or not isinstance(terminal, str) or not terminal:
        return _local_error("outcome_unknown", "operation journal omitted operation_id or terminal", status=status)
    if operation_id != requested_operation_id:
        # The route itself identifies the requested journal.  Accepting a
        # different body receipt would let a stale or misrouted response make
        # the caller act on another workflow's state.
        return _operation_outcome_unknown(
            status, requested_operation_id, "operation journal returned a mismatched operation receipt"
        )
    if status == 202:
        if terminal not in {"pending", "unknown"}:
            return _local_error("outcome_unknown", "accepted operation is not pending or unknown", status=status)
        outcome = "pending"
    else:
        if terminal not in {"succeeded", "failed"}:
            return _local_error("outcome_unknown", "terminal operation has an unknown state", status=status)
        outcome = "completed"
    return _ok({"outcome": outcome, "operation_id": operation_id, "state": terminal, "operation": parsed})


def _operation_outcome_unknown(status, requested_id, message):
    """Keep a safe requested receipt id available for a later status lookup."""
    result = {
        "ok": False,
        "error": {"code": "outcome_unknown", "message": message},
        "status": status,
    }
    if isinstance(requested_id, str) and requested_id:
        result["operation_id"] = requested_id
    return json.dumps(result, ensure_ascii=False)


def _mutation_operation_outcome(status, request_body, parsed):
    """Every operation-enabled mutation response must prove its journal receipt."""
    requested = (request_body or {}).get("operation")
    receipt = parsed.get("operation") if isinstance(parsed, dict) else None
    requested_id = requested.get("operation_id") if isinstance(requested, dict) else None
    if not isinstance(requested, dict) or not isinstance(receipt, dict):
        return _operation_outcome_unknown(status, requested_id, "mutation omitted operation receipt")
    operation_id = receipt.get("operation_id")
    terminal = receipt.get("terminal")
    expected_terminals = {202: {"pending", "unknown"}, 200: {"succeeded", "failed"}}
    if not isinstance(requested_id, str) or not requested_id or requested_id != operation_id:
        return _operation_outcome_unknown(status, requested_id, "mutation returned a mismatched operation receipt")
    if terminal not in expected_terminals.get(status, set()):
        return _operation_outcome_unknown(status, requested_id, "mutation returned an operation receipt with an invalid terminal")
    outcome = "pending" if status == 202 else "completed"
    return _ok({"outcome": outcome, "operation_id": operation_id, "state": terminal, "operation": receipt})


def _record_app_operation_attempt(args, result_json):
    """ADIC v1: log every app_operation outcome, tagged with its operation
    name, to the active turn-scoped ledger (see
    gateway.session_context.record_import_attempt).

    Deliberately NOT filtered to the literal "data.import" here: local-server
    stamps job["import_operation"] with the APP's own declared write
    operation name (e.g. "records.refresh" for a blueprint app), not a fixed
    string. cron/scheduler.py does the name filtering at verdict time against
    that per-job value. Pre-filtering by a hardcoded name here would silently
    stop recording for any app whose write operation isn't literally named
    "data.import" — every round would then read an empty ledger and judge
    the job a hard failure, which is the mirror image of the bug this
    workstream exists to fix (false success flipped into false failure).

    Every other action — including call(), whose app-level errors are
    deliberately surfaced as ok:true (two-layer status) so the interactive
    model can self-correct — is left completely untouched by this function.
    Outside a cron run no scope is open, so this is a no-op: interactive
    behavior does not change at all.
    """
    if str(args.get("action", "") or "").strip() != "app_operation":
        return
    operation = str(args.get("app_operation", "") or "").strip()
    if not operation:
        return
    try:
        parsed = json.loads(result_json)
        if not isinstance(parsed, dict):
            return
        ok = parsed.get("ok") is True
        error_code, error_message = "", ""
        if not ok:
            error = parsed.get("error")
            if isinstance(error, dict):
                error_code = str(error.get("code", "") or "")
                error_message = str(error.get("message", "") or "")
            elif error is not None:
                error_message = str(error)
        from gateway.session_context import record_import_attempt
        record_import_attempt(
            operation=operation, ok=ok, error_code=error_code, error_message=error_message
        )
    except Exception:
        # Bookkeeping must never break the tool response the model is
        # waiting on.
        pass


def app_host_tool(args, **_kw):
    # Tool handlers must return a STRING (json-encoded) — a raw dict reaches
    # the model provider as non-string content and gets rejected (same
    # contract as list_my_channels).
    args = args or {}
    result = _app_host_tool_dispatch(args, **_kw)
    _record_app_operation_attempt(args, result)
    return result


def _app_host_tool_dispatch(args, **_kw):
    action = str(args.get("action", "") or "").strip()

    if action == "build_env":
        # Pure local check — no HTTP request, no credentials leave the tool.
        return _build_env_result()

    try:
        method, path, body, timeout = _build_request(action, args)
        execution_headers = _execution_headers()
        scoped_auto_refresh_token = _auto_refresh_scope_token(
            action, body, execution_headers
        )
    except _BadRequest as exc:
        return _local_error("invalid_request", str(exc), status=_STATUS_NOT_SENT)
    except _AutoRefreshScopeUnavailable as exc:
        return _local_error(
            "automatic_maintenance_unavailable", str(exc), status=_STATUS_NOT_SENT
        )

    base = _base_url()
    token = _secret("ZETTLAB_AGENT_ACTION_TOKEN")
    if not base or not token:
        return _local_error(
            "unsupported",
            "App Host 未配置或不可用，这台设备暂不支持生成应用",
            status=_STATUS_NOT_SENT,
        )

    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {
        _ACTION_TOKEN_HEADER: scoped_auto_refresh_token or token,
        "Accept": "application/json",
    }
    headers.update(execution_headers)
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)

    try:
        with _urlopen(req, timeout=timeout) as resp:
            status = resp.status
            # Raw header, NOT headers.get_content_type(): that helper answers
            # a default of text/plain when the server sent no Content-Type at
            # all (a real 204), which would make "genuinely content-free" and
            # "declared text" indistinguishable.
            content_type = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Same read cap as the success path — an oversized error body must
        # not balloon memory on a 2 GB shared device.
        upstream = _parse_upstream_error(exc.read(_MAX_RESPONSE_BYTES + 1) or b"")
        if upstream is not None:
            # Verbatim pass-through: skills branch on the upstream `code`
            # string (slug_conflict / storage_full / ...), never the HTTP
            # status. Do not flatten into prose.
            return _fail(upstream, status=exc.code)
        if action in ("rollback", "publish", "call") and exc.code == 404:
            # Hermes and local-server ship as separate OTA packages, so this
            # tool can meet a server that predates POST /{slug}/rollback. Its
            # router answers an unregistered path with a bodiless 404, while
            # every business 404 on this face (unknown app / not the owner)
            # carries a JSON {code} body and took the verbatim branch above —
            # so a rollback 404 WITHOUT a parsable body means the route does
            # not exist. Reporting it as transport_error would invite retries
            # of a request that can never work; "unsupported" is terminal and
            # the message names the fallback that always exists.
            if action == "call":
                # No fallback exists for call — unlike publish/rollback there
                # is no older channel that reaches an app's endpoints. The
                # message must break the "404 means the app is missing"
                # reading, or the model wanders off probing other slugs.
                message = (
                    "设备端 App Host 尚不支持 call（local-server 版本较旧）。"
                    "这不代表应用不存在——是这台设备没有转发通道，"
                    "换路径、换 slug 或重试都不会成功，请如实汇报此能力缺失"
                )
            elif action == "publish":
                message = (
                    "设备端 App Host 尚不支持 publish（local-server 版本较旧）。"
                    "改用老设备发布通道：把本次 source_subdir 指向的那个目录"
                    "（metadata.json 就在它下面那一层）整个拷到 App Host 的 "
                    ".staging 下作为其直接子目录——拷完 metadata.json 必须正好在 "
                    ".staging/<新目录>/ 里，不能再套一层——再调 install（新建）"
                    "或 reload（修改）"
                )
            else:
                message = (
                    "设备端 App Host 尚不支持 rollback（local-server 版本较旧）。"
                    "请改用 reload 以旧内容重新构建来完成撤销"
                )
            return _local_error("unsupported", message, status=exc.code)
        return _local_error(
            "transport_error",
            f"App Host 请求失败（HTTP {exc.code}），未返回可解析的错误体",
            status=exc.code,
        )
    except Exception:
        # Never echo the exception: URLError/timeout messages can embed the
        # request URL. status=None is deliberate — the request DID go out and
        # the outcome is unknown, so the caller may resend idempotently.
        return _local_error("transport_error", "无法连接 App Host 服务", status=None)

    if len(raw) > _MAX_RESPONSE_BYTES:
        return _local_error("transport_error", "App Host 返回内容过大", status=status)

    # Each action has a frozen completion code. A generic 2xx acceptance would
    # wrongly report an asynchronous 202 as completed.
    # deliberately answers 204 with no body (release_slot always; delete is
    # idempotent, a retried DELETE must also get 204), and logs answers 2xx
    # with text/plain. Treating "2xx but body isn't JSON" as transport_error
    # reported every successful release/delete as a failure. The tier test is
    # the 2xx range, never an enumeration of specific codes.
    if status in _COMPLETION_STATUS.get(action, set()):
        text = raw.decode("utf-8", errors="replace")
        if action in {"publish", "install", "reload"} and isinstance((body or {}).get("operation"), dict):
            if not text.strip() or "json" not in content_type:
                return _operation_outcome_unknown(
                    status, body["operation"].get("operation_id"),
                    "mutation returned no JSON receipt",
                )
            try:
                return _mutation_operation_outcome(status, body, json.loads(text))
            except Exception:
                return _operation_outcome_unknown(
                    status, body["operation"].get("operation_id"),
                    "mutation returned invalid JSON",
                )
        if action in {"workflow_operation_status", "workflow_operation_resume"}:
            if not text.strip() or "json" not in content_type:
                return _local_error("outcome_unknown", "operation journal returned no JSON receipt", status=status)
            try:
                return _workflow_operation_status(
                    status, json.loads(text), str(args.get("operation_id", "") or "").strip()
                )
            except Exception:
                return _local_error("outcome_unknown", "operation journal returned invalid JSON", status=status)
        if content_type.startswith("text/"):
            # Declared text (logs): the response SHAPE follows the declared
            # type, never the accident of emptiness — a fresh app's empty log
            # still answers with the full text-payload contract
            # (text/truncated/total_chars), which the skill reads without
            # existence checks.
            return _ok(_text_payload(text))
        if not text.strip():
            # Genuinely content-free answers (204 from release_slot/delete
            # carry no Content-Type at all): nothing to shape.
            return _ok({})
        if "json" in content_type:
            try:
                return _ok(json.loads(text))
            except Exception:
                # Declared JSON but unparseable: still a 2xx success at the
                # HTTP layer — hand the raw text back rather than erroring.
                return _ok(_text_payload(text))
        return _ok(_text_payload(text))

    # A response arrived, but it is not a completion for this action. This is
    # especially important for asynchronous mutation routes: callers must
    # query the returned receipt instead of retrying or claiming completion.
    return _local_error(
        "outcome_unknown", f"App Host 返回了非完成状态（HTTP {status}）", status=status
    )


from tools.registry import registry  # noqa: E402

registry.register(
    name="app_host",
    toolset="zettlab_apphost",
    schema=APP_HOST_SCHEMA,
    handler=app_host_tool,
    check_fn=_check_app_host,
    emoji="🏗️",
    # App Host is the platform-native entry point for creating and managing
    # generated apps. Keep its schema directly visible when Tool Search is
    # enabled, matching app_data; the toolset and scope gates still decide
    # whether it is available at all.
    defer_to_tool_search=False,
)
