import base64
import io
import json
import urllib.error
from unittest.mock import patch

from tests.tools._profile_scope import mux_profile_scope
from tools.app_workspace_tool import APP_WORKSPACE_SCHEMA, app_workspace_tool


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
        "status", "checkout", "read", "apply_patch", "build", "publish",
        "discard", "maintainer_schedule_status", "maintainer_schedule",
    }
    assert not {"command", "url", "host", "env", "shell", "directory"} & set(props)
    assert "relative" in props["path"]["description"]


def test_workspace_routes_and_wire_shapes(monkeypatch):
    cases = [
        ("status", {}, "GET", f"/{_SLUG}/workspace?expected_instance_id={_INSTANCE}", None, {"checked_out": False}),
        ("checkout", {}, "POST", f"/{_SLUG}/workspace/checkout", {"expected_instance_id": _INSTANCE}, {"checked_out": True}),
        ("read", {"path": "main.go"}, "POST", f"/{_SLUG}/workspace/read", {"expected_instance_id": _INSTANCE, "path": "main.go"}, {"path": "main.go", "content": base64.b64encode(b"package main").decode(), "sha256": _SHA}),
        ("build", {}, "POST", f"/{_SLUG}/workspace/build", {"expected_instance_id": _INSTANCE}, {"revision": 4}),
        ("publish", {"expected_revision": 4, "note": "Fix title"}, "POST", f"/{_SLUG}/workspace/publish", {"expected_instance_id": _INSTANCE, "expected_revision": 4, "note": "Fix title"}, {"version_id": "v2"}),
        ("maintainer_schedule_status", {}, "GET", f"/{_SLUG}/maintainer_schedule?expected_instance_id={_INSTANCE}", None, {"app_instance_id": _INSTANCE, "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": False, "schedule_revision": 4}),
        ("maintainer_schedule", {"expected_schedule_revision": 4, "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": False}, "POST", f"/{_SLUG}/maintainer_schedule", {"expected_instance_id": _INSTANCE, "expected_schedule_revision": 4, "schedule": "0 9 * * *", "timezone": "Asia/Shanghai", "enabled": False}, {"app_instance_id": _INSTANCE, "schedule_revision": 5}),
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


def test_workspace_rejects_generic_path_and_unused_fields_without_request(monkeypatch):
    def never(*_):
        raise AssertionError("invalid request must not be sent")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        path_result = json.loads(app_workspace_tool(_args("read", path="../metadata.json")))
        extra_result = json.loads(app_workspace_tool(_args("build", command="go test ./...")))
    assert path_result["status"] == 0 and path_result["error"]["code"] == "invalid_request"
    assert extra_result["status"] == 0 and extra_result["error"]["code"] == "invalid_request"


def test_bodyless_old_server_404_fails_closed_without_fallback(monkeypatch):
    error = urllib.error.HTTPError(_BASE + "/x", 404, "not found", {}, io.BytesIO(b"404 page not found"))
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", side_effect=error):
        output = json.loads(app_workspace_tool(_args("checkout")))
    assert output["ok"] is False
    assert output["error"]["code"] == "unsupported"
    assert output["status"] == 404


def test_workspace_rejects_wrong_success_status(monkeypatch):
    with mux_profile_scope(monkeypatch, _SCOPE), patch(
        "tools.app_workspace_tool._apphost._urlopen", _capture({}, _Response({"checked_out": True}, status=202))
    ):
        output = json.loads(app_workspace_tool(_args("checkout")))
    assert output["ok"] is False
    assert output["error"]["code"] == "outcome_unknown"


def test_maintainer_schedule_requires_its_own_revision_and_verified_receipt(monkeypatch):
    def never(*_):
        raise AssertionError("invalid schedule request must not be sent")
    with mux_profile_scope(monkeypatch, _SCOPE), patch("tools.app_workspace_tool._apphost._urlopen", never):
        missing = json.loads(app_workspace_tool(_args(
            "maintainer_schedule", schedule="0 9 * * *", timezone="Asia/Shanghai", enabled=True,
        )))
        old_field = json.loads(app_workspace_tool(_args(
            "maintainer_schedule", expected_revision=4, schedule="0 9 * * *", timezone="Asia/Shanghai", enabled=True,
        )))
    assert missing["error"]["code"] == "invalid_request" and missing["status"] == 0
    assert old_field["error"]["code"] == "invalid_request" and old_field["status"] == 0

    with mux_profile_scope(monkeypatch, _SCOPE), patch(
        "tools.app_workspace_tool._apphost._urlopen",
        _capture({}, _Response({"app_instance_id": _INSTANCE, "schedule_revision": 5})),
    ):
        accepted = json.loads(app_workspace_tool(_args(
            "maintainer_schedule", expected_schedule_revision=4,
            schedule="0 9 * * *", timezone="Asia/Shanghai", enabled=True,
        )))
    assert accepted["ok"] is True

    for bad in (
        {"app_instance_id": _INSTANCE},
        {"app_instance_id": "other", "schedule_revision": 5},
        {"app_instance_id": _INSTANCE, "schedule_revision": -1},
    ):
        with mux_profile_scope(monkeypatch, _SCOPE), patch(
            "tools.app_workspace_tool._apphost._urlopen", _capture({}, _Response(bad))
        ):
            rejected = json.loads(app_workspace_tool(_args(
                "maintainer_schedule", expected_schedule_revision=4,
                schedule="0 9 * * *", timezone="Asia/Shanghai", enabled=True,
            )))
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "outcome_unknown"


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
