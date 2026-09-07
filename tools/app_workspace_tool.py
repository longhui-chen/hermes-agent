"""Narrow adapter for an app's dedicated-maintainer workspace and task surface.

This is intentionally not a terminal, generic cron, or filesystem bridge.
App Host owns the checkout, validates the dedicated maintainer binding, and
allows maintenance tasks only for that current app instance.  Each task is
bound to a declared app mutation capability rather than a raw URL or database.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode

from tools import apphost_tool as _apphost
from tools.registry import registry


_ACTIONS = frozenset({
    "status", "checkout", "list", "read", "apply_patch", "build", "publish",
    "discard", "maintainer_schedule_status", "maintenance_tasks",
    "create_maintenance_task", "update_maintenance_task",
    "delete_maintenance_task", "maintenance_task_runs",
})
_MAX_RESPONSE_BYTES = 1024 * 1024
# App Host accepts an 8 MiB patch, but a subsequent read serializes its
# ``[]byte`` content as base64 while this adapter intentionally caps every
# response at 1 MiB.  Keep a 10 KiB wire-envelope margin (path + JSON) below
# that cap so every accepted UTF-8 replacement can be read back by this tool.
_MAX_PATCH_BYTES = 760 << 10
_MAX_PATH_CHARS = 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


APP_WORKSPACE_SCHEMA = {
    "name": "app_workspace",
    "description": (
        "Edit the current version of an app only through App Host's dedicated "
        "maintainer workspace. This is a fixed checkout/list/read/replace/build/"
        "publish/discard surface, not a terminal, general filesystem, URL, or "
        "environment interface. Start with status, use its app_instance_id and "
        "revision as the required compare-and-swap values, and read a file "
        "before replacing it with apply_patch. After checkout, use list to see "
        "which files exist and read only paths it returned — never guess a "
        "pathname. A generated app keeps its page source at static/index.html, "
        "its server code in main.go and any schema in migrations/, but list is "
        "the authority; a read that comes back absent means your path was "
        "wrong, not that the app lacks that kind of source. The one exception: "
        "if list itself fails with error.code list_unsupported, this device's "
        "App Host predates the list route — every other action still works, so "
        "fall back to probing with read and do guess pathnames there; that "
        "error is never evidence the app or its source is missing."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ACTIONS),
                "description": "One fixed App Workspace action.",
            },
            "slug": {
                "type": "string",
                "description": "The app slug whose dedicated maintainer owns this workspace.",
            },
            "expected_instance_id": {
                "type": "string",
                "description": "Required current app_instance_id from status; prevents acting on a replaced app.",
            },
            "path": {
                "type": "string",
                "description": "For read/apply_patch only: a relative regular-file path inside this checked-out app workspace; never an absolute host path.",
            },
            "expected_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
                "description": "For apply_patch only: SHA-256 returned by the immediately preceding read.",
            },
            "content": {
                "type": "string",
                "description": "For apply_patch only: complete UTF-8 replacement file content. App Host receives it as its typed byte field; no patch command or shell is exposed.",
            },
            "expected_revision": {
                "type": "integer",
                "minimum": 0,
                "description": "For publish only: app workspace revision returned by status, preventing a stale checkout from publishing.",
            },
            "note": {
                "type": "string",
                "description": "For publish only: a concise user-facing description of this version change.",
            },
            "name": {"type": "string", "description": "For create_maintenance_task: user-visible task name."},
            "schedule": {"type": "string", "description": "For create_maintenance_task: recurring interval or cron expression."},
            "timezone": {"type": "string", "description": "For create_maintenance_task: IANA timezone."},
            "kind": {"type": "string", "enum": ["refresh", "summary"], "description": "For create_maintenance_task: the app-scoped maintenance kind."},
            "app_operation": {"type": "string", "description": "For create_maintenance_task: declared app write operation, read from app_capabilities first."},
            "capability_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$", "description": "For create_maintenance_task: exact digest from app_capabilities for app_operation."},
            "instruction": {"type": "string", "description": "For create_maintenance_task: concise user-approved collection or summary instruction."},
            "task_id": {"type": "string", "description": "For update_maintenance_task, delete_maintenance_task, or maintenance_task_runs: the id returned by maintenance_tasks."},
            "expected_schedule_revision": {"type": "integer", "minimum": 0, "description": "For update_maintenance_task: current schedule_revision returned by maintenance_tasks."},
            "enabled": {"type": "boolean", "description": "For update_maintenance_task: whether this task should run."},
        },
        "required": ["action", "slug", "expected_instance_id"],
    },
}


def _check_app_workspace():
    return bool(_apphost._base_url() and _apphost._secret("ZETTLAB_AGENT_ACTION_TOKEN"))


_check_app_workspace._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _bad_request(message: str) -> str:
    return _apphost._local_error("invalid_request", message, status=_apphost._STATUS_NOT_SENT)


def _required_instance(args: dict) -> str:
    instance = str(args.get("expected_instance_id", "") or "").strip()
    if not instance or len(instance) > 256 or any(ord(ch) < 0x20 for ch in instance):
        raise _apphost._BadRequest("expected_instance_id 必须是合法的当前 app_instance_id")
    return instance


def _required_path(args: dict) -> str:
    path = str(args.get("path", "") or "").strip()
    if not path or path == "." or path.startswith("/") or "\\" in path:
        raise _apphost._BadRequest("path 必须是 app workspace 内的相对文件路径")
    if len(path) > _MAX_PATH_CHARS:
        raise _apphost._BadRequest("path 超过 App Workspace 上限")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _apphost._BadRequest("path 必须是 app workspace 内的相对文件路径")
    return path


def _required_revision(args: dict) -> int:
    revision = args.get("expected_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise _apphost._BadRequest("expected_revision 必须是 status 返回的非负整数")
    return revision


def _required_task_id(args: dict) -> str:
    value = str(args.get("task_id", "") or "").strip()
    if not value or len(value) > 256 or value in {".", ".."} or "/" in value or "\\" in value:
        raise _apphost._BadRequest("task_id 必须来自当前应用的 maintenance_tasks")
    return value


def _required_schedule_revision(args: dict) -> int:
    value = args.get("expected_schedule_revision")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _apphost._BadRequest("expected_schedule_revision 必须来自 maintenance_tasks")
    return value


def _only(args: dict, allowed: set[str]):
    unexpected = set(args) - allowed
    if unexpected:
        raise _apphost._BadRequest("请求包含不属于此 action 的字段")


def _build_request(args: dict):
    action = args.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise _apphost._BadRequest("action 必须是声明的 App Workspace action")
    slug = _apphost._require_slug(args)
    instance = _required_instance(args)
    root = f"/{quote(slug, safe='')}/workspace"
    base_fields = {"action", "slug", "expected_instance_id"}
    if action == "status":
        _only(args, base_fields)
        return "GET", root + "?" + urlencode({"expected_instance_id": instance}), None, _apphost._DEFAULT_TIMEOUT
    if action == "maintainer_schedule_status":
        _only(args, base_fields)
        return "GET", f"/{quote(slug, safe='')}/maintainer_schedule?" + urlencode({
            "expected_instance_id": instance,
        }), None, _apphost._DEFAULT_TIMEOUT
    if action == "maintenance_tasks":
        _only(args, base_fields)
        return "GET", f"/{quote(slug, safe='')}/maintenance_tasks?" + urlencode({"expected_instance_id": instance}), None, _apphost._DEFAULT_TIMEOUT
    if action == "create_maintenance_task":
        fields = base_fields | {"name", "schedule", "timezone", "kind", "app_operation", "capability_digest", "instruction"}
        _only(args, fields)
        required = ("name", "schedule", "timezone", "kind", "app_operation", "capability_digest", "instruction")
        if any(not isinstance(args.get(key), str) or not str(args[key]).strip() for key in required):
            raise _apphost._BadRequest("create_maintenance_task requires its declared task contract")
        digest = str(args["capability_digest"]).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise _apphost._BadRequest("capability_digest must come from app_capabilities")
        return "POST", f"/{quote(slug, safe='')}/maintenance_tasks", {
            "expected_instance_id": instance, "name": str(args["name"]).strip(), "schedule": str(args["schedule"]).strip(),
            "timezone": str(args["timezone"]).strip(), "kind": str(args["kind"]).strip(), "app_operation": str(args["app_operation"]).strip(),
            "capability_digest": digest, "instruction": str(args["instruction"]).strip(),
        }, _apphost._DEFAULT_TIMEOUT
    if action == "maintenance_task_runs":
        _only(args, base_fields | {"task_id"})
        return "GET", f"/{quote(slug, safe='')}/maintenance_tasks/{quote(_required_task_id(args), safe='')}/runs?" + urlencode({"expected_instance_id": instance}), None, _apphost._DEFAULT_TIMEOUT
    if action == "delete_maintenance_task":
        _only(args, base_fields | {"task_id", "expected_schedule_revision"})
        return "DELETE", f"/{quote(slug, safe='')}/maintenance_tasks/{quote(_required_task_id(args), safe='')}?" + urlencode({
            "expected_instance_id": instance, "expected_schedule_revision": _required_schedule_revision(args),
        }), None, _apphost._DEFAULT_TIMEOUT
    if action == "update_maintenance_task":
        fields = base_fields | {"task_id", "expected_schedule_revision", "name", "schedule", "timezone", "kind", "app_operation", "capability_digest", "instruction", "enabled"}
        _only(args, fields)
        required = ("name", "schedule", "timezone", "kind", "app_operation", "capability_digest", "instruction")
        if any(not isinstance(args.get(key), str) or not str(args[key]).strip() for key in required) or not isinstance(args.get("enabled"), bool):
            raise _apphost._BadRequest("update_maintenance_task requires the complete task contract and enabled state")
        digest = str(args["capability_digest"]).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise _apphost._BadRequest("capability_digest must come from app_capabilities")
        return "PATCH", f"/{quote(slug, safe='')}/maintenance_tasks/{quote(_required_task_id(args), safe='')}", {
            "expected_instance_id": instance, "expected_schedule_revision": _required_schedule_revision(args),
            "name": str(args["name"]).strip(), "schedule": str(args["schedule"]).strip(), "timezone": str(args["timezone"]).strip(),
            "kind": str(args["kind"]).strip(), "app_operation": str(args["app_operation"]).strip(), "capability_digest": digest,
            "instruction": str(args["instruction"]).strip(), "enabled": args["enabled"],
        }, _apphost._DEFAULT_TIMEOUT
    if action in {"checkout", "build", "discard"}:
        _only(args, base_fields)
        method = "DELETE" if action == "discard" else "POST"
        path = root if action == "discard" else f"{root}/{action}"
        return method, path, {"expected_instance_id": instance}, _apphost._LONG_TIMEOUT if action == "build" else _apphost._DEFAULT_TIMEOUT
    if action == "list":
        _only(args, base_fields)
        return "POST", root + "/list", {"expected_instance_id": instance}, _apphost._DEFAULT_TIMEOUT
    if action == "read":
        _only(args, base_fields | {"path"})
        return "POST", root + "/read", {
            "expected_instance_id": instance, "path": _required_path(args),
        }, _apphost._DEFAULT_TIMEOUT
    if action == "apply_patch":
        _only(args, base_fields | {"path", "expected_sha256", "content"})
        expected_sha256 = str(args.get("expected_sha256", "") or "").strip()
        content = args.get("content")
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise _apphost._BadRequest("apply_patch 需要 read 返回的 expected_sha256")
        if not isinstance(content, str):
            raise _apphost._BadRequest("apply_patch 需要 string 类型的完整 content")
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > _MAX_PATCH_BYTES:
            raise _apphost._BadRequest("apply_patch content 超过 App Workspace 上限")
        return "POST", root + "/apply_patch", {
            "expected_instance_id": instance,
            "path": _required_path(args),
            "expected_sha256": expected_sha256.lower(),
            # Go's []byte JSON encoding is base64. The model-facing contract
            # remains UTF-8 source text; only this typed adapter performs the
            # wire conversion.
            "content": base64.b64encode(content_bytes).decode("ascii"),
        }, _apphost._DEFAULT_TIMEOUT
    if action == "publish":
        _only(args, base_fields | {"expected_revision", "note"})
        note = args.get("note", "")
        if not isinstance(note, str):
            raise _apphost._BadRequest("publish note 必须是 string")
        return "POST", root + "/publish", {
            "expected_instance_id": instance,
            "expected_revision": _required_revision(args),
            "note": note,
        }, _apphost._LONG_TIMEOUT
    raise AssertionError("declared workspace actions were handled above")


def _normalize_read(parsed):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
        return parsed
    try:
        content = base64.b64decode(parsed["content"], validate=True)
        text = content.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        # The server permits regular files, not only UTF-8 source. Preserve
        # its exact base64 shape in that rare case instead of corrupting it.
        return parsed
    result = dict(parsed)
    result["content"] = text
    return result


def _schedule_response(parsed, *, expected_instance_id: str, is_status: bool):
    """Validate App Host's narrow schedule resource before exposing it."""
    if not isinstance(parsed, dict):
        return None
    instance = parsed.get("app_instance_id")
    revision = parsed.get("schedule_revision")
    if (
        not isinstance(instance, str)
        or not instance
        or instance != expected_instance_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
    ):
        return None
    if is_status and (
        not isinstance(parsed.get("schedule"), str)
        or not isinstance(parsed.get("timezone"), str)
        or not isinstance(parsed.get("enabled"), bool)
    ):
        return None
    return parsed


def _maintenance_task_response(parsed, *, action: str):
    """Reject a partial success receipt before an Agent claims a task changed."""
    if action == "maintenance_tasks":
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tasks"), list):
            return None
        for task in parsed["tasks"]:
            if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
                return None
        return parsed
    if action in {"create_maintenance_task", "update_maintenance_task"}:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("id"), str) or not parsed["id"]:
            return None
        return parsed
    if action == "maintenance_task_runs":
        if not isinstance(parsed, dict) or not isinstance(parsed.get("occurrences"), list):
            return None
        return parsed
    return parsed


def app_workspace_tool(args, **_kw) -> str:
    args = args if isinstance(args, dict) else {}
    try:
        method, path, body, timeout = _build_request(args)
    except _apphost._BadRequest as exc:
        return _bad_request(str(exc))

    action = args.get("action")
    base = _apphost._base_url()
    token = _apphost._secret("ZETTLAB_AGENT_ACTION_TOKEN")
    if not base or not token:
        return _apphost._local_error("unsupported", "当前 Agent 未配置 App Workspace", status=_apphost._STATUS_NOT_SENT)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {_apphost._ACTION_TOKEN_HEADER: token, "Accept": "application/json"}
    headers.update(_apphost._execution_headers())
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with _apphost._urlopen(request, timeout=timeout) as response:
            status = response.status
            content_type = (response.headers.get("Content-Type") or "").lower()
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raw = exc.read(_MAX_RESPONSE_BYTES + 1) or b""
        upstream = _apphost._parse_upstream_error(raw)
        if upstream is not None:
            return _apphost._fail(upstream, status=exc.code)
        if exc.code == 404:
            # list 是这一批里唯一的新路由：Hermes 先于 App Host 部署时，只有它
            # 会撞 404，而 status / read / apply_patch / build / publish 在旧
            # 服务端上全都照常可用。若也回「尚不支持 App Workspace」，维护者会
            # 把一个路由缺失读成整个工作区不可用而放弃整轮维护——工具说明里还
            # 写着「先 list 再 read」，它更没有理由继续。所以这一档单独降级，
            # 并直接告诉它替代走法。
            if action == "list":
                return _apphost._local_error(
                    "list_unsupported",
                    "这台设备的 App Host 版本还没有列文件能力。**只有列文件这一个动作缺失**，"
                    "status / read / apply_patch / build / publish 全都照常可用，工作区也已经检出，"
                    "不要据此判断应用不存在、源码找不到或维护无法继续。"
                    "改用逐个 read 探路：若这是本 skill 生成的应用，先试 static/index.html（页面）、"
                    "main.go（后端）、migrations/ 下的 .sql（建表）——这几条只是生成应用的**候选**，"
                    "blueprint 应用或用户自己调整过目录结构时它们可能都不在。任何一个路径 read 不到，"
                    "只说明这个文件不存在，换一个继续试；这种情况下允许按应用类型推测路径，"
                    "「先 list 再 read」那条要求不适用于这台设备。",
                    status=exc.code,
                )
            return _apphost._local_error(
                "unsupported",
                "设备端 App Host 尚不支持 App Workspace；没有安全的兼容路径",
                status=exc.code,
            )
        return _apphost._local_error("transport_error", f"App Workspace 请求失败（HTTP {exc.code}），未返回可解析的错误体", status=exc.code)
    except Exception:
        return _apphost._local_error("transport_error", "无法连接 App Workspace 服务", status=None)
    if len(raw) > _MAX_RESPONSE_BYTES:
        return _apphost._local_error("transport_error", "App Workspace 返回内容过大", status=status)
    if action in {"apply_patch", "discard", "delete_maintenance_task"}:
        if status == 204 and not raw:
            return _apphost._ok({})
        return _apphost._local_error("outcome_unknown", "App Workspace 返回了非合同完成状态", status=status)
    if status != 200 or "json" not in content_type:
        return _apphost._local_error("outcome_unknown", "App Workspace 返回了非合同完成状态", status=status)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _apphost._local_error("outcome_unknown", "App Workspace 返回了无效 JSON", status=status)
    if action == "maintainer_schedule_status":
        checked = _schedule_response(
            parsed,
            expected_instance_id=str(args.get("expected_instance_id") or "").strip(),
            is_status=True,
        )
        if checked is None:
            return _apphost._local_error(
                "outcome_unknown",
                "maintainer schedule 返回了缺失或不匹配的实例/修订回执",
                status=status,
            )
        return _apphost._ok(checked)
    if action in {"maintenance_tasks", "create_maintenance_task", "update_maintenance_task", "maintenance_task_runs"}:
        checked = _maintenance_task_response(parsed, action=action)
        if checked is None:
            return _apphost._local_error("outcome_unknown", "maintenance task 返回了不完整的回执", status=status)
        return _apphost._ok(checked)
    return _apphost._ok(_normalize_read(parsed) if action == "read" else parsed)


registry.register(
    name="app_workspace",
    toolset="zettlab_app_workspace",
    schema=APP_WORKSPACE_SCHEMA,
    handler=app_workspace_tool,
    check_fn=_check_app_workspace,
    emoji="workspace",
    max_result_size_chars=_MAX_RESPONSE_BYTES,
    defer_to_tool_search=False,
)
