import base64
import io
import json
import urllib.error
from unittest.mock import patch

from tests.tools._profile_scope import mux_profile_scope
from tools.app_workspace_tool import (
    APP_WORKSPACE_SCHEMA,
    _MAX_PATCH_BYTES,
    _MAX_PATH_CHARS,
    app_workspace_tool,
)


_BASE = "http://127.0.0.1:18080/api/v1/internal/apphost"
_SCOPE = {
    "ZET_APPHOST_BASE_URL": _BASE,
    "ZETTLAB_AGENT_ACTION_TOKEN": "workspace-token",
}
_SLUG = "workspace-app"
_INSTANCE = "instance-1"
_SHA = "a" * 64


class _Headers:
    def __init__(self, content_type="application/json"):
        self.content_type = content_type

    def get(self, name, default=None):
        return self.content_type if name.lower() == "content-type" else default


class _Response:
    def __init__(self, payload=None, *, status=200, content_type="application/json"):
        self.status = status
        self.headers = _Headers(content_type)
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode()


def _capture(seen, response):
    def open_request(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        return response
    return open_request


def _args(action, **extra):
    return {"action": action, "slug": _SLUG, "expected_instance_id": _INSTANCE, **extra}


def test_schema_is_fixed_workspace_surface_not_generic_host_access():
    props = APP_WORKSPACE_SCHEMA["parameters"]["properties"]
    assert set(props["action"]["enum"]) == {
        "status", "checkout", "list", "read", "apply_patch", "build", "build_status",
        "publish", "discard", "maintainer_schedule_status", "maintenance_tasks",
        "create_maintenance_task", "update_maintenance_task",
        "delete_maintenance_task", "maintenance_task_runs",
        "maintenance_task_events",
    }
    assert not {"command", "url", "host", "env", "shell", "directory"} & set(props)
    assert "relative" in props["path"]["description"]


def test_schema_tells_the_maintainer_to_list_before_reading():
    # A maintainer asked to reword an app's page guessed seven pathnames, never
    # tried static/index.html, and reported the app had no page source. The
    # surface has to say out loud that listing is how you find a path, and that
    # an absent file means a wrong guess rather than a missing capability.
    description = APP_WORKSPACE_SCHEMA["description"]
    assert "list" in description
    assert "never guess a pathname" in description
    assert "static/index.html" in description
    assert "your path was wrong" in description


def test_workspace_routes_and_wire_shapes(monkeypatch):
    cases = [
        ("status", {}, "GET", f"/{_SLUG}/workspace?expected_instance_id={_INSTANCE}", None, {"checked_out": False}),
        ("checkout", {}, "POST", f"/{_SLUG}/workspace/checkout", {"expected_instance_id": _INSTANCE}, {"checked_out": True}),
        ("list", {}, "POST", f"/{_SLUG}/workspace/list", {"expected_instance_id": _INSTANCE}, {"files": [{"path": "main.go", "size": 12}, {"path": "static/index.html", "size": 5668}]}),
        ("read", {"path": "main.go"}, "POST", f"/{_SLUG}/workspace/read", {"expected_instance_id": _INSTANCE, "path": "main.go"}, {"path": "main.go", "content": base64.b64encode(b"package main").decode(), "sha256": _SHA}),
        ("build", {}, "POST", f"/{_SLUG}/workspace/build", {"expected_instance_id": _INSTANCE}, {"revision": 4}),
        ("publish", {"expected_revision": 4, "note": "Fix title"}, "POST", f"/{_SLUG}/workspace/publish", {"expected_instance_id": _INSTANCE, "expected_revision": 4, "note": "Fix title"}, {"version_id": "v2"}),
        ("maintainer_schedule_status", {}, "GET", f"/{_SLUG}/maintainer_schedule?expected_instance_id={_INSTANCE}", None, {"app_instance_id": _INSTANCE, "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": False, "schedule_revision": 4}),
        ("maintenance_tasks", {}, "GET", f"/{_SLUG}/maintenance_tasks?expected_instance_id={_INSTANCE}", None, {"tasks": []}),
        ("create_maintenance_task", {"name": "Hourly summary", "schedule": "every 30m", "timezone": "Asia/Shanghai", "kind": "summary", "app_operation": "weather.summarize", "capability_digest": _SHA, "instruction": "Summarize the latest weather data."}, "POST", f"/{_SLUG}/maintenance_tasks", {"expected_instance_id": _INSTANCE, "name": "Hourly summary", "schedule": "every 30m", "timezone": "Asia/Shanghai", "kind": "summary", "app_operation": "weather.summarize", "capability_digest": _SHA, "instruction": "Summarize the latest weather data."}, {"id": "job-2"}),
        ("maintenance_task_runs", {"task_id": "job-2"}, "GET", f"/{_SLUG}/maintenance_tasks/job-2/runs?expected_instance_id={_INSTANCE}", None, {"occurrences": []}),
        ("update_maintenance_task", {"task_id": "job-2", "expected_schedule_revision": 4, "name": "Hourly summary", "schedule": "every 30m", "timezone": "Asia/Shanghai", "kind": "summary", "app_operation": "weather.summarize", "capability_digest": _SHA, "instruction": "Summarize the latest weather data.", "enabled": False}, "PATCH", f"/{_SLUG}/maintenance_tasks/job-2", {"expected_instance_id": _INSTANCE, "expected_schedule_revision": 4, "name": "Hourly summary", "schedule": "every 30m", "timezone": "Asia/Shanghai", "kind": "summary", "app_operation": "weather.summarize", "capability_digest": _SHA, "instruction": "Summarize the latest weather data.", "enabled": False}, {"id": "job-2"}),
    ]
    with mux_profile_scope(monkeypatch, _SCOPE):
        for action, extra, method, path, expected_body, response in cases:
            seen = {}
            with patch("tools.app_workspace_tool._apphost._urlopen", _capture(seen, _Response(response))):
                output = json.loads(app_workspace_tool(_args(action, **extra)))
            assert output["ok"] is True
            assert seen["request"].method == method
            assert seen["request"].full_url == _BASE + path
            body = seen["request"].data
            assert (json.loads(body) if body else None) == expected_body
            if action == "read":
                assert output["data"]["content"] == "package main"
            if action == "list":
                # The nested page source must survive the round trip: it is the
                # exact path the real maintainer never managed to guess.
                assert [f["path"] for f in output["data"]["files"]] == ["main.go", "static/index.html"]


def test_apply_patch_encodes_text_as_server_byte_wire_and_discard_requires_204(monkeypatch):
    with mux_profile_scope(monkeypatch, _SCOPE):
        seen = {}
        with patch("tools.app_workspace_tool._apphost._urlopen", _capture(seen, _Response(None, status=204, content_type=None))):
            output = json.loads(app_workspace_tool(_args(
                "apply_patch", path="main.go", expected_sha256=_SHA, content="package main\n",
            )))
        assert output == {"ok": True, "data": {}}
        assert seen["request"].full_url == _BASE + f"/{_SLUG}/workspace/apply_patch"
        assert json.loads(seen["request"].data) == {
            "expected_instance_id": _INSTANCE,
            "path": "main.go",
            "expected_sha256": _SHA,
            "content": base64.b64encode(b"package main\n").decode(),
        }
        seen = {}
        with patch("tools.app_workspace_tool._apphost._urlopen", _capture(seen, _Response(None, status=204, content_type=None))):
            output = json.loads(app_workspace_tool(_args("discard")))
        assert output == {"ok": True, "data": {}}
        assert seen["request"].method == "DELETE"
        assert seen["request"].full_url == _BASE + f"/{_SLUG}/workspace"
        seen = {}
        with patch("tools.app_workspace_tool._apphost._urlopen", _capture(seen, _Response(None, status=204, content_type=None))):
            output = json.loads(app_workspace_tool(_args("delete_maintenance_task", task_id="job-2", expected_schedule_revision=4)))
        assert output == {"ok": True, "data": {}}
        assert seen["request"].method == "DELETE"
        assert seen["request"].full_url == _BASE + f"/{_SLUG}/maintenance_tasks/job-2?expected_instance_id={_INSTANCE}&expected_schedule_revision=4"


def test_workspace_rejects_generic_path_and_unused_fields_without_request(monkeypatch):
    def never(*_):
        raise AssertionError("invalid request must not be sent")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        path_result = json.loads(app_workspace_tool(_args("read", path="../metadata.json")))
        extra_result = json.loads(app_workspace_tool(_args("build", command="go test ./...")))
    assert path_result["status"] == 0 and path_result["error"]["code"] == "invalid_request"
    assert extra_result["status"] == 0 and extra_result["error"]["code"] == "invalid_request"


def test_apply_patch_rejects_content_that_cannot_fit_a_follow_up_read(monkeypatch):
    def never(*_):
        raise AssertionError("unreadable replacement must not be sent")

    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        output = json.loads(app_workspace_tool(_args(
            "apply_patch", path="main.go", expected_sha256=_SHA,
            content="x" * (_MAX_PATCH_BYTES + 1),
        )))
    assert output["status"] == 0
    assert output["error"]["code"] == "invalid_request"


def test_workspace_rejects_path_that_would_exhaust_the_read_response_margin(monkeypatch):
    def never(*_):
        raise AssertionError("oversized path must not be sent")

    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        output = json.loads(app_workspace_tool(_args("read", path="a" * (_MAX_PATH_CHARS + 1))))
    assert output["status"] == 0
    assert output["error"]["code"] == "invalid_request"


def test_bodyless_old_server_404_fails_closed_without_fallback(monkeypatch):
    error = urllib.error.HTTPError(_BASE + "/x", 404, "not found", {}, io.BytesIO(b"404 page not found"))
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", side_effect=error):
        output = json.loads(app_workspace_tool(_args("checkout")))
    assert output["ok"] is False
    assert output["error"]["code"] == "unsupported"
    assert output["status"] == 404


def test_list_404_on_older_server_degrades_to_a_usable_path(monkeypatch):
    # 部署顺序问题：Hermes 先于 App Host 上线时，只有 list 这个新路由会 404，
    # 而 status / read / apply_patch / build / publish 在旧服务端上照常可用。
    # 若也回「尚不支持 App Workspace」，维护者会把一个路由缺失读成整个工作区
    # 不可用而放弃整轮维护——工具说明里还写着「先 list 再 read」，它更没有理由
    # 继续。所以这一档必须单独降级，并且要给出可以照做的替代路径。
    error = urllib.error.HTTPError(_BASE + "/x", 404, "not found", {}, io.BytesIO(b"404 page not found"))
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", side_effect=error):
        output = json.loads(app_workspace_tool(_args("list")))
    assert output["ok"] is False
    assert output["status"] == 404

    # 复审抓到的要害：**降级必须落在 error.code 上，不能只落在文案上**。上层
    # skill 按 code 分支（PUBLISH.md 明写 `error.code="unsupported"` = 设备不支持、
    # 告知用户并停止），沿用 `unsupported` 的话，调用方在读到这段新文案之前就已经
    # 走完终止路径了，动作级降级等于没做。已有 `publish_unsupported` 这个先例。
    assert output["error"]["code"] == "list_unsupported", (
        "list 路由缺失必须有自己的 code，不能跟「整个 App Workspace 不支持」共用 unsupported"
    )

    message = output["error"]["message"]
    assert "只有列文件这一个动作缺失" in message, "不能让维护者以为整个工作区不可用"
    assert "static/index.html" in message and "main.go" in message, "必须给出可以直接照做的替代路径"
    # 第二条复审 finding：这几条路径只对本 skill 生成的应用成立。blueprint 应用或
    # 用户自己调过目录结构时它们可能都不在，文案不能把它们说成确定答案——否则维护者
    # 三条都读不到，又受「never guess a pathname」约束，仍会判定源码找不到而放弃。
    assert "候选" in message, "必须说明这几条路径只是生成应用的候选，不是确定答案"
    assert "不适用于这台设备" in message, "必须解除「先 list 再 read」的约束，否则维护者无路可走"

    # 其它动作的 404 仍然是「整个 App Workspace 不支持」，语义不能被这次改动冲淡
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", side_effect=error):
        other = json.loads(app_workspace_tool(_args("checkout")))
    assert other["error"]["code"] == "unsupported"
    assert "只有列文件这一个动作缺失" not in other["error"]["message"]


def test_tool_description_tells_the_maintainer_how_to_recover_from_list_unsupported(monkeypatch):
    # 工具说明里写着「never guess a pathname」。若不在同一处说明 list_unsupported
    # 是唯一例外，维护者会在旧服务端上被自己的工具说明锁死：既拿不到 list，又不
    # 允许猜路径。约束必须落在它实际生效的那份文本里。
    description = APP_WORKSPACE_SCHEMA["description"]
    assert "list_unsupported" in description
    assert "guess pathnames" in description


def test_workspace_rejects_wrong_success_status(monkeypatch):
    with mux_profile_scope(monkeypatch, _SCOPE), patch(
        "tools.app_workspace_tool._apphost._urlopen", _capture({}, _Response({"checked_out": True}, status=202))
    ):
        output = json.loads(app_workspace_tool(_args("checkout")))
    assert output["ok"] is False
    assert output["error"]["code"] == "outcome_unknown"


def test_maintainer_schedule_mutation_is_not_exposed_without_a_confirmation_contract(monkeypatch):
    def never(*_):
        raise AssertionError("unconfirmed schedule request must not be sent")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        output = json.loads(app_workspace_tool(_args(
            "maintainer_schedule", expected_schedule_revision=4,
            schedule="0 9 * * *", timezone="Asia/Shanghai", enabled=True,
        )))
    assert output["error"]["code"] == "invalid_request"
    assert output["status"] == 0


def test_maintainer_schedule_status_rejects_absent_or_incomplete_revision(monkeypatch):
    for bad in (
        {"app_instance_id": _INSTANCE, "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": True},
        {"app_instance_id": _INSTANCE, "schedule_revision": 4},
        {"app_instance_id": "other", "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": True, "schedule_revision": 4},
    ):
        with mux_profile_scope(monkeypatch, _SCOPE), patch(
            "tools.app_workspace_tool._apphost._urlopen", _capture({}, _Response(bad))
        ):
            output = json.loads(app_workspace_tool(_args("maintainer_schedule_status")))
        assert output["ok"] is False
        assert output["error"]["code"] == "outcome_unknown"


def _sequence(seen, responses):
    """_urlopen stand-in that answers one queued response per request and
    records every request it saw (in order)."""
    queue = list(responses)

    def open_request(request, timeout=None):
        seen.setdefault("requests", []).append(request)
        if not queue:
            raise AssertionError("more requests than scripted responses")
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    return open_request


def test_build_starts_the_background_job_and_polls_until_terminal(monkeypatch):
    # The device queues the rebuild behind another build; the tool must not
    # start a second one, must poll the status route, and must hand the
    # maintainer the queue position while it waits.
    monkeypatch.setattr("tools.app_workspace_tool._BUILD_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr("tools.app_workspace_tool._BUILD_WAIT_BUDGET_SECONDS", 30.0)
    seen = {}
    responses = [
        _Response({"build_id": "wb-1", "state": "queued", "queue_position": 2, "detail": "排队等待设备空闲（前面还有 2 个构建）", "attempt": 1, "max_attempts": 4}, status=202),
        _Response({"build_id": "wb-1", "state": "building", "detail": "正在编译", "attempt": 1, "max_attempts": 4}),
        _Response({"build_id": "wb-1", "state": "succeeded", "detail": "构建完成，可以发布", "attempt": 1, "max_attempts": 4, "fingerprint": "f" * 64}),
    ]
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, responses)):
        output = json.loads(app_workspace_tool(_args("build")))
    assert output["ok"] is True
    assert output["data"]["state"] == "succeeded"
    assert "publish" in output["data"]["next"]
    methods = [(r.method, r.full_url) for r in seen["requests"]]
    assert methods[0] == ("POST", f"{_BASE}/{_SLUG}/workspace/build")
    assert methods[1:] == [("GET", f"{_BASE}/{_SLUG}/workspace/build?expected_instance_id={_INSTANCE}")] * 2
    assert json.loads(seen["requests"][0].data) == {"expected_instance_id": _INSTANCE}


def test_build_hands_back_a_running_job_with_a_next_that_forbids_rebuilding(monkeypatch):
    monkeypatch.setattr("tools.app_workspace_tool._BUILD_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr("tools.app_workspace_tool._BUILD_WAIT_BUDGET_SECONDS", 0.0)
    seen = {}
    responses = [
        _Response({"build_id": "wb-2", "state": "queued", "queue_position": 3, "detail": "排队等待设备空闲（前面还有 3 个构建）", "attempt": 1, "max_attempts": 4}, status=202),
    ]
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, responses)):
        output = json.loads(app_workspace_tool(_args("build")))
    assert output["ok"] is True
    assert output["data"]["state"] == "queued"
    assert output["data"]["queue_position"] == 3
    assert "前面还有 3 个构建" in output["data"]["next"]
    assert "build_status" in output["data"]["next"]
    assert "不要重复 build" in output["data"]["next"]
    assert len(seen["requests"]) == 1


def test_build_on_an_older_synchronous_server_is_treated_as_finished(monkeypatch):
    seen = {}
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response({"revision": 4})])):
        output = json.loads(app_workspace_tool(_args("build")))
    assert output["ok"] is True
    assert output["data"]["state"] == "succeeded"
    assert output["data"]["revision"] == 4
    assert len(seen["requests"]) == 1


def test_build_status_route_and_failure_next(monkeypatch):
    seen = {}
    failed = {"build_id": "wb-3", "state": "failed", "failure_kind": "compile_failed", "message": "编译失败，见 log", "log": "main.go:3:1: undefined: err", "attempt": 1, "max_attempts": 4}
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response(failed)])):
        output = json.loads(app_workspace_tool(_args("build_status", build_id="wb-3")))
    assert output["ok"] is True
    assert output["data"]["state"] == "failed"
    assert "apply_patch" in output["data"]["next"]
    request = seen["requests"][0]
    assert request.method == "GET"
    assert request.full_url == f"{_BASE}/{_SLUG}/workspace/build/wb-3?expected_instance_id={_INSTANCE}"

    # Without an id the current build is asked for.
    seen = {}
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response(dict(failed, state="retry_wait", failure_kind="device_busy", detail="设备忙；1 分钟 后自动重试（第 2/4 次）"))])):
        output = json.loads(app_workspace_tool(_args("build_status")))
    assert output["ok"] is True
    assert seen["requests"][0].full_url == f"{_BASE}/{_SLUG}/workspace/build?expected_instance_id={_INSTANCE}"
    assert "build_status" in output["data"]["next"]


def test_publish_is_refused_locally_while_the_last_build_is_not_succeeded(monkeypatch):
    seen = {}
    running = {"build_id": "wb-4", "state": "building", "detail": "正在编译", "attempt": 1, "max_attempts": 4}
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response(running)])):
        output = json.loads(app_workspace_tool(_args("publish", expected_revision=4, note="Fix title")))
    assert output["ok"] is False
    assert output["error"]["code"] == "build_in_progress"
    assert output["status"] == 0
    assert [r.method for r in seen["requests"]] == ["GET"], "publish must not have been sent"

    seen = {}
    failed = dict(running, state="failed", failure_kind="compile_failed", log="x.go:1:1: syntax error")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response(failed)])):
        output = json.loads(app_workspace_tool(_args("publish", expected_revision=4, note="Fix title")))
    assert output["ok"] is False
    assert output["error"]["code"] == "build_required"

    # A succeeded build lets publish through; an older server without the
    # status route (bodyless 404) is left to judge for itself.
    seen = {}
    ok = dict(running, state="succeeded")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [_Response(ok), _Response({"version_id": "v2"})])):
        output = json.loads(app_workspace_tool(_args("publish", expected_revision=4, note="Fix title")))
    assert output["ok"] is True and output["data"]["version_id"] == "v2"
    assert [r.method for r in seen["requests"]] == ["GET", "POST"]
    seen = {}
    missing = urllib.error.HTTPError(f"{_BASE}/x", 404, "Not Found", {}, io.BytesIO(b""))
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", _sequence(seen, [missing, _Response({"version_id": "v3"})])):
        output = json.loads(app_workspace_tool(_args("publish", expected_revision=4, note="Fix title")))
    assert output["ok"] is True and output["data"]["version_id"] == "v3"
