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
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode

from tools import apphost_tool as _apphost
from tools.registry import registry


_ACTIONS = frozenset({
    "status", "checkout", "list", "read", "apply_patch", "sync_capabilities",
    "build", "build_status",
    "publish", "discard", "maintainer_schedule_status", "maintenance_tasks",
    "create_maintenance_task", "update_maintenance_task",
    "delete_maintenance_task", "maintenance_task_runs",
    "maintenance_task_events", "register_maintenance_tasks_from_spec",
})
_MAX_RESPONSE_BYTES = 1024 * 1024
# The rebuild is a background job on App Host (it queues behind other builds
# on a device whose memory fits one). ``build`` starts it and then polls for a
# bounded while so a fast build still answers in one call; a slower one is
# handed back as queued/building with a ``next`` that says to poll
# ``build_status`` — never to start another build.
_BUILD_POLL_INTERVAL_SECONDS = 5.0
_BUILD_WAIT_BUDGET_SECONDS = 90.0
_BUILD_TERMINAL_STATES = frozenset({"succeeded", "failed", "interrupted"})
_BUILD_RETRY_KINDS = frozenset({"device_busy", "out_of_memory", "timeout", "cache_invalid"})
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
        "build_status/publish/discard surface, not a terminal, general filesystem, "
        "URL, or environment interface. Start with status, use its app_instance_id "
        "and revision as the required compare-and-swap values, and read a file "
        "before replacing it with apply_patch. build starts a background rebuild "
        "and waits a short while: state succeeded means go on to publish; state "
        "queued/building/waiting_memory/retry_wait means the device is busy — tell "
        "the user where it stands (queue_position / detail), then poll build_status "
        "until it is terminal; never call build again while one is running. "
        "publish requires the latest build to have succeeded on the current "
        "content (error codes build_required / build_stale: build again). "
        "sync_capabilities installs, from the platform presets, the capability "
        "files the workspace spec.yaml declares (capabilities.ai -> appai.go "
        "with platformAISummarize/platformAIChoose; cron/events/agent_tasks -> "
        "apptasks.go) and grants them in metadata; run it after editing "
        "spec.yaml capabilities and before build. The platform always offers a "
        "stateless AI face to apps — never tell the user no AI model is "
        "available; declare ai in spec.yaml, sync_capabilities, then call the "
        "appai.go helpers. A 404 from sync_capabilities means this device's "
        "App Host predates the route: say so and stop, do not hand-copy files. "
        "After checkout, use list to see "
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
                "description": (
                    "One fixed App Workspace action. register_maintenance_tasks_from_spec "
                    "registers the workspace spec.yaml's task declarations and is TWO calls: "
                    "call it without confirm to get the plan (create / update / delete), show "
                    "the user that plan with every deletion named, and only after an explicit "
                    "yes call it again with confirm=true and the same plan_digest."
                ),
            },
            "slug": {
                "type": "string",
                "description": "The app slug whose dedicated maintainer owns this workspace.",
            },
            "expected_instance_id": {
                "type": "string",
                "description": "Current app_instance_id from status (or the app list's app_instance_id); required for every action except status, which is how you obtain it. Prevents acting on a replaced app.",
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
            "build_id": {
                "type": "string",
                "description": "For build_status only (optional): the build_id returned by build; omitted means this workspace's current build.",
            },
            "note": {
                "type": "string",
                "description": "For publish only: a concise user-facing description of this version change.",
            },
            "name": {"type": "string", "description": "For create_maintenance_task: user-visible task name."},
            "schedule": {"type": "string", "description": "For create_maintenance_task with trigger=schedule (the default): recurring interval or cron expression. Omit for event/manual tasks — the platform wakes them."},
            "timezone": {"type": "string", "description": "For create_maintenance_task with trigger=schedule: IANA timezone."},
            "trigger": {"type": "string", "enum": ["schedule", "event", "manual"], "description": "For create_maintenance_task: what wakes the task. Default schedule. event = a platform event fires; manual = the app asks for it."},
            "event": {"type": "string", "description": "For create_maintenance_task with trigger=event: the platform event name. It must be in the platform's event dictionary — read it from the app-coding skill's PLATFORM_EVENTS.md; a name that is not in it is refused (the error lists the valid ones)."},
            "event_path": {"type": "string", "description": "For create_maintenance_task with trigger=event: the absolute directory this subscription watches. Required only for the events the dictionary marks as directory-scoped, and refused for the others."},
            "kind": {"type": "string", "enum": ["refresh", "summary"], "description": "For create_maintenance_task: the app-scoped maintenance kind."},
            "app_operation": {"type": "string", "description": "For create_maintenance_task: declared app write operation, read from app_capabilities first."},
            "capability_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$", "description": "For create_maintenance_task: exact digest from app_capabilities for app_operation."},
            "instruction": {"type": "string", "description": "For create_maintenance_task: concise user-approved collection or summary instruction."},
            "task_id": {"type": "string", "description": "For update_maintenance_task, delete_maintenance_task, maintenance_task_runs, or maintenance_task_events: the id returned by maintenance_tasks."},
            "spec_id": {"type": "string", "description": "For maintenance_task_events: the task's spec id (as named in the run prompt) when task_id is unknown."},
            "creation_key": {"type": "string", "description": "For maintenance_task_events: the handle a task created at runtime cites instead of a spec id — its own run prompt names it (creation_key=…). Use it when the prompt gave you no spec_id."},
            "ack": {"type": "boolean", "description": "For maintenance_task_events: acknowledge (remove) the returned events; default true. Pass false to peek."},
            "confirm": {"type": "boolean", "description": "For register_maintenance_tasks_from_spec: false/omitted returns the plan only (nothing is changed). Pass true ONLY after showing the user the plan — deletions named one by one — and getting an explicit yes."},
            "expected_plan_digest": {"type": "string", "description": "For register_maintenance_tasks_from_spec with confirm=true: the plan_digest of the plan the user agreed to. A mismatch is refused: fetch the plan again and re-confirm."},
            "expected_schedule_revision": {"type": "integer", "minimum": 0, "description": "For update_maintenance_task: current schedule_revision returned by maintenance_tasks."},
            "enabled": {"type": "boolean", "description": "For update_maintenance_task: whether this task should run."},
        },
        "required": ["action", "slug"],
    },
}


def _check_app_workspace():
    return bool(_apphost._base_url() and _apphost._secret("ZETTLAB_AGENT_ACTION_TOKEN"))


_check_app_workspace._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _bad_request(message: str) -> str:
    return _apphost._local_error("invalid_request", message, status=_apphost._STATUS_NOT_SENT)


def _optional_instance(args: dict) -> str:
    """expected_instance_id 可缺省（只有 status 用）；给了就按 _required_instance 的规则校验。"""
    raw = args.get("expected_instance_id")
    if raw is None or str(raw).strip() == "":
        return ""
    return _required_instance(args)


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
    root = f"/{quote(slug, safe='')}/workspace"
    base_fields = {"action", "slug", "expected_instance_id"}
    if action == "status":
        # 起点动作：维护者就是从 status 拿 app_instance_id 的，所以它可以不带；
        # 带了照旧透传（服务端做 CAS）。以前这里也强制要，鸡生蛋——工具永远起不了步。
        _only(args, base_fields)
        instance = _optional_instance(args)
        query = "?" + urlencode({"expected_instance_id": instance}) if instance else ""
        return "GET", root + query, None, _apphost._DEFAULT_TIMEOUT
    instance = _required_instance(args)
    if action == "maintainer_schedule_status":
        _only(args, base_fields)
        return "GET", f"/{quote(slug, safe='')}/maintainer_schedule?" + urlencode({
            "expected_instance_id": instance,
        }), None, _apphost._DEFAULT_TIMEOUT
    if action == "maintenance_tasks":
        _only(args, base_fields)
        return "GET", f"/{quote(slug, safe='')}/maintenance_tasks?" + urlencode({"expected_instance_id": instance}), None, _apphost._DEFAULT_TIMEOUT
    if action == "create_maintenance_task":
        fields = base_fields | {"name", "schedule", "timezone", "kind", "app_operation", "capability_digest", "instruction", "trigger", "event", "event_path"}
        _only(args, fields)
        # 触发方式：不传＝定时（老调用方一个字不用改）。事件 / 按需任务由平台唤醒，
        # 没有自己的节奏，所以不要 schedule / timezone；事件名与「哪个事件要目录
        # 作用域」由服务端按平台事件字典判，这里不复制一份名单（两套口径会打架）。
        trigger = str(args.get("trigger", "") or "").strip().lower() or "schedule"
        if trigger not in {"schedule", "event", "manual"}:
            raise _apphost._BadRequest("trigger 只能是 schedule / event / manual")
        required = ["name", "kind", "app_operation", "capability_digest", "instruction"]
        if trigger == "schedule":
            required += ["schedule", "timezone"]
        if any(not isinstance(args.get(key), str) or not str(args[key]).strip() for key in required):
            raise _apphost._BadRequest("create_maintenance_task requires its declared task contract")
        event = str(args.get("event", "") or "").strip()
        event_path = str(args.get("event_path", "") or "").strip()
        if trigger == "event" and not event:
            raise _apphost._BadRequest("trigger=event 必须写 event（平台事件名，见 app-coding 技能的 PLATFORM_EVENTS.md）")
        if trigger != "event" and (event or event_path):
            raise _apphost._BadRequest("只有 trigger=event 的任务才写 event / event_path")
        digest = str(args["capability_digest"]).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise _apphost._BadRequest("capability_digest must come from app_capabilities")
        body = {
            "expected_instance_id": instance, "name": str(args["name"]).strip(),
            "schedule": str(args.get("schedule", "") or "").strip(), "timezone": str(args.get("timezone", "") or "").strip(),
            "kind": str(args["kind"]).strip(), "app_operation": str(args["app_operation"]).strip(),
            "capability_digest": digest, "instruction": str(args["instruction"]).strip(),
            "trigger": trigger,
        }
        if trigger == "event":
            body["event"] = event
            if event_path:
                body["event_path"] = event_path
        return "POST", f"/{quote(slug, safe='')}/maintenance_tasks", body, _apphost._DEFAULT_TIMEOUT
    if action == "register_maintenance_tasks_from_spec":
        # 两拍：不带 confirm 只算计划（服务端不落地任何东西），带 confirm + 同一份
        # plan_digest 才执行。技能里要求先把计划、尤其删除项念给用户听。
        _only(args, base_fields | {"confirm", "expected_plan_digest"})
        confirm = args.get("confirm", False)
        if not isinstance(confirm, bool):
            raise _apphost._BadRequest("confirm 必须是布尔值")
        digest = str(args.get("expected_plan_digest", "") or "").strip().lower()
        if confirm and not _SHA256_RE.fullmatch(digest):
            raise _apphost._BadRequest("confirm=true 必须带上给用户看过的那份计划的 plan_digest")
        if not confirm and digest:
            raise _apphost._BadRequest("expected_plan_digest 只在 confirm=true 时用")
        body = {"expected_instance_id": instance, "confirm": confirm}
        if confirm:
            body["expected_plan_digest"] = digest
        return "POST", f"/{quote(slug, safe='')}/maintenance_tasks/register_from_spec", body, _apphost._DEFAULT_TIMEOUT
    if action == "maintenance_task_events":
        _only(args, base_fields | {"task_id", "spec_id", "creation_key", "ack"})
        query = {"expected_instance_id": instance}
        task_id = str(args.get("task_id", "") or "").strip()
        spec_id = str(args.get("spec_id", "") or "").strip()
        # 运行期建出来的任务没有 spec_id：它的唤醒提示词给的是 creation_key
        # （提示词写在 hermes 之外，由 App Host 生成，这里只负责把句柄透过去）。
        creation_key = str(args.get("creation_key", "") or "").strip()
        if task_id:
            query["task_id"] = _required_task_id(args)
        elif spec_id:
            if len(spec_id) > 64 or "/" in spec_id or "\\" in spec_id:
                raise _apphost._BadRequest("spec_id 必须是任务的 spec id")
            query["spec_id"] = spec_id
        elif creation_key:
            if not _SHA256_RE.fullmatch(creation_key):
                raise _apphost._BadRequest("creation_key 必须照抄本任务唤醒提示词里的那一串")
            query["creation_key"] = creation_key.lower()
        else:
            raise _apphost._BadRequest("maintenance_task_events 需要 task_id、spec_id 或 creation_key（照抄本任务唤醒提示词里给的那个）")
        ack = args.get("ack", True)
        if not isinstance(ack, bool):
            raise _apphost._BadRequest("ack 必须是布尔值")
        query["ack"] = "true" if ack else "false"
        return "GET", f"/{quote(slug, safe='')}/maintenance_task_events?" + urlencode(query), None, _apphost._DEFAULT_TIMEOUT
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
    if action == "build_status":
        _only(args, base_fields | {"build_id"})
        build_id = str(args.get("build_id", "") or "").strip()
        if build_id and (len(build_id) > 64 or "/" in build_id or "\\" in build_id):
            raise _apphost._BadRequest("build_id 必须是 build 返回的 build_id")
        suffix = f"/{quote(build_id, safe='')}" if build_id else ""
        return "GET", f"{root}/build{suffix}?" + urlencode({"expected_instance_id": instance}), None, _apphost._DEFAULT_TIMEOUT
    if action in {"checkout", "build", "discard", "sync_capabilities"}:
        _only(args, base_fields)
        method = "DELETE" if action == "discard" else "POST"
        path = root if action == "discard" else f"{root}/{action}"
        return method, path, {"expected_instance_id": instance}, _apphost._DEFAULT_TIMEOUT
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
    if action == "maintenance_task_events":
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            return None
        return parsed
    if action == "register_maintenance_tasks_from_spec":
        # 没有 plan_digest / applied 就不是这个动作的回执：Agent 绝不能凭一个说不清
        # 的 200 去跟用户说「已按规格登记」。三张清单也要在，缺一张读的人会以为
        # 「没有要删的」，而其实是服务端没回答。
        if not isinstance(parsed, dict) or not isinstance(parsed.get("plan_digest"), str) or not isinstance(parsed.get("applied"), bool):
            return None
        if any(not isinstance(parsed.get(key), list) for key in ("create", "update", "delete")):
            return None
        return parsed
    return parsed


class _WireFailure(Exception):
    """A request that did not produce a usable 2xx JSON body; ``envelope`` is
    the tool's failure envelope for it."""

    def __init__(self, envelope: str):
        super().__init__(envelope)
        self.envelope = envelope


def _send(base: str, token: str, method: str, path: str, body, timeout, *, action: str):
    """One App Workspace request. Returns (status, content_type, raw) or raises
    _WireFailure carrying the finished failure envelope."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {_apphost._ACTION_TOKEN_HEADER: token, "Accept": "application/json"}
    headers.update(_apphost._execution_headers())
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with _apphost._urlopen(request, timeout=timeout) as response:
            return response.status, (response.headers.get("Content-Type") or "").lower(), response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raw = exc.read(_MAX_RESPONSE_BYTES + 1) or b""
        upstream = _apphost._parse_upstream_error(raw)
        if upstream is not None:
            raise _WireFailure(_apphost._fail(upstream, status=exc.code))
        if exc.code == 404:
            if action == "list":
                raise _WireFailure(_apphost._local_error(
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
                ))
            if action == "build_status":
                raise _WireFailure(_apphost._local_error(
                    "build_status_unsupported",
                    "这台设备的 App Host 还没有后台构建状态路由：它的 build 是同步完成的，"
                    "build 返回 ok 即已编译完成，直接 publish；不要再查 build_status。",
                    status=exc.code,
                ))
            raise _WireFailure(_apphost._local_error(
                "unsupported",
                "设备端 App Host 尚不支持 App Workspace；没有安全的兼容路径",
                status=exc.code,
            ))
        raise _WireFailure(_apphost._local_error("transport_error", f"App Workspace 请求失败（HTTP {exc.code}），未返回可解析的错误体", status=exc.code))
    except _WireFailure:
        raise
    except Exception:
        raise _WireFailure(_apphost._local_error("transport_error", "无法连接 App Workspace 服务", status=None))


def _decode_json(status, content_type, raw, *, accept=(200,)):
    """Return the parsed body of a JSON 2xx, or the failure envelope string."""
    if len(raw) > _MAX_RESPONSE_BYTES:
        return None, _apphost._local_error("transport_error", "App Workspace 返回内容过大", status=status)
    if status not in accept or "json" not in content_type:
        return None, _apphost._local_error("outcome_unknown", "App Workspace 返回了非合同完成状态", status=status)
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _apphost._local_error("outcome_unknown", "App Workspace 返回了无效 JSON", status=status)


def _build_next(state: dict) -> str:
    """The one line that tells the maintainer what to do with a build state."""
    st = str(state.get("state") or "")
    detail = str(state.get("detail") or "")
    if st == "succeeded":
        return "构建成功；接着 publish（带 status 返回的 expected_revision）。"
    if st in {"failed", "interrupted"}:
        kind = str(state.get("failure_kind") or "")
        if kind == "compile_failed":
            return "编译失败：按 log 里的报错改源码（read → apply_patch），再 build。"
        if kind in _BUILD_RETRY_KINDS:
            return "设备侧失败（不是代码问题），平台已自动重试过；如实告诉用户设备忙/内存紧张，稍后再 build 一次。"
        if kind == "interrupted":
            return "构建被中断（服务重启或工作区被丢弃）；重新 build 一次。"
        return "构建失败，见 message；不是代码问题时如实告诉用户。"
    ahead = state.get("queue_position")
    where = f"排队中（前面还有 {ahead} 个构建）" if st == "queued" and isinstance(ahead, int) and ahead > 0 else (detail or f"构建进行中（{st}）")
    return (
        f"{where}。构建在后台继续，不要重复 build：先把这个进度告诉用户，"
        "稍后调 action=build_status（同一个 slug / expected_instance_id）查看，"
        "等 state=succeeded 再 publish。"
    )


def _finish_build(parsed, status):
    """Shape the build/build_status answer: the server's state plus ``next``."""
    if not isinstance(parsed, dict):
        return _apphost._local_error("outcome_unknown", "App Workspace 返回了不完整的构建状态", status=status)
    if "state" not in parsed:
        # An older App Host builds synchronously and answers with the
        # workspace status: reaching here means the build already finished.
        result = dict(parsed)
        result["state"] = "succeeded"
        result["next"] = _build_next(result)
        return _apphost._ok(result)
    result = dict(parsed)
    result["next"] = _build_next(result)
    return _apphost._ok(result)


def _run_build(base, token, slug, instance, *, method, path, body, timeout):
    """Start the background rebuild, then poll it for a bounded while."""
    try:
        status, content_type, raw = _send(base, token, method, path, body, timeout, action="build")
    except _WireFailure as exc:
        return exc.envelope
    parsed, envelope = _decode_json(status, content_type, raw, accept=(200, 202))
    if envelope is not None:
        return envelope
    if not isinstance(parsed, dict) or "state" not in parsed or str(parsed.get("state")) in _BUILD_TERMINAL_STATES:
        return _finish_build(parsed, status)
    deadline = time.monotonic() + _BUILD_WAIT_BUDGET_SECONDS
    status_path = f"/{quote(slug, safe='')}/workspace/build?" + urlencode({"expected_instance_id": instance})
    latest = parsed
    while time.monotonic() < deadline:
        time.sleep(min(_BUILD_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
        try:
            status, content_type, raw = _send(base, token, "GET", status_path, None, _apphost._DEFAULT_TIMEOUT, action="build_status")
        except _WireFailure:
            break  # keep the last state we saw; the maintainer polls build_status
        polled, envelope = _decode_json(status, content_type, raw)
        if envelope is not None or not isinstance(polled, dict):
            break
        latest = polled
        if str(polled.get("state")) in _BUILD_TERMINAL_STATES:
            break
    return _finish_build(latest, 200)


def _last_build_blocks_publish(base, token, slug, instance):
    """Publish is refused here, before any request, when the workspace's last
    build is known not to have succeeded; the server enforces the same."""
    path = f"/{quote(slug, safe='')}/workspace/build?" + urlencode({"expected_instance_id": instance})
    try:
        status, content_type, raw = _send(base, token, "GET", path, None, _apphost._DEFAULT_TIMEOUT, action="build_status")
    except _WireFailure:
        return None  # an older server (or no build yet): let the server judge
    parsed, envelope = _decode_json(status, content_type, raw)
    if envelope is not None or not isinstance(parsed, dict) or "state" not in parsed:
        return None
    st = str(parsed.get("state"))
    if st == "succeeded":
        return None
    result = dict(parsed)
    result["next"] = _build_next(result)
    if st in _BUILD_TERMINAL_STATES:
        return _apphost._fail({"code": "build_required", "message": f"最近一次构建 {st}（{result.get('failure_kind') or ''}），不能发布。" + result["next"]}, status=_apphost._STATUS_NOT_SENT)
    return _apphost._fail({"code": "build_in_progress", "message": "构建还在进行中，不能发布。" + result["next"]}, status=_apphost._STATUS_NOT_SENT)


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
    if action == "build":
        return _run_build(base, token, str(args.get("slug")), str(args.get("expected_instance_id") or "").strip(),
                          method=method, path=path, body=body, timeout=timeout)
    if action == "publish":
        blocked = _last_build_blocks_publish(base, token, str(args.get("slug")), str(args.get("expected_instance_id") or "").strip())
        if blocked is not None:
            return blocked
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
    if action in {"maintenance_tasks", "create_maintenance_task", "update_maintenance_task", "maintenance_task_runs", "register_maintenance_tasks_from_spec"}:
        checked = _maintenance_task_response(parsed, action=action)
        if checked is None:
            return _apphost._local_error("outcome_unknown", "maintenance task 返回了不完整的回执", status=status)
        return _apphost._ok(checked)
    if action == "build_status":
        return _finish_build(parsed, status)
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
