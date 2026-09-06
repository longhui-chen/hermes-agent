"""sync_capabilities：维护者改了 spec.yaml 的 capabilities 后，让 App Host 从 presets 装能力件
（appai.go / apptasks.go）并授权——工具只把动作按固定形状转给服务端，不自己碰文件。"""

import json
from unittest.mock import patch

from tests.tools._profile_scope import mux_profile_scope
from tools.app_workspace_tool import APP_WORKSPACE_SCHEMA, app_workspace_tool


_BASE = "http://127.0.0.1:18080/api/v1/internal/apphost"
_SCOPE = {"ZET_APPHOST_BASE_URL": _BASE, "ZETTLAB_AGENT_ACTION_TOKEN": "workspace-token"}
_SLUG = "workspace-app"
_INSTANCE = "instance-1"


class _Headers:
    def get(self, name, default=None):
        return "application/json" if name.lower() == "content-type" else default


class _Response:
    def __init__(self, payload, status=200):
        self.status = status
        self.headers = _Headers()
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *_):
        return json.dumps(self._payload).encode("utf-8")


def _capture(seen, response):
    def _urlopen(request, timeout=None):
        seen["request"] = request
        return response
    return _urlopen


def test_sync_capabilities_is_in_the_fixed_surface():
    assert "sync_capabilities" in APP_WORKSPACE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "sync_capabilities" in APP_WORKSPACE_SCHEMA["description"]
    assert "never tell the user no AI model is available" in APP_WORKSPACE_SCHEMA["description"]


def test_sync_capabilities_posts_instance_cas_and_returns_report(monkeypatch):
    report = {"ok": True, "mode": "sync-capabilities", "added": ["appai.go"], "granted": ["app_ai"],
              "already": ["events.go"], "warnings": []}
    seen = {}
    with mux_profile_scope(monkeypatch, _SCOPE):
        with patch("tools.app_workspace_tool._apphost._urlopen", _capture(seen, _Response(report))):
            output = json.loads(app_workspace_tool({
                "action": "sync_capabilities", "slug": _SLUG, "expected_instance_id": _INSTANCE,
            }))
    assert output["ok"] is True
    assert seen["request"].method == "POST"
    assert seen["request"].full_url == _BASE + f"/{_SLUG}/workspace/sync_capabilities"
    assert json.loads(seen["request"].data) == {"expected_instance_id": _INSTANCE}
    assert output["data"]["added"] == ["appai.go"] and output["data"]["granted"] == ["app_ai"]


def test_sync_capabilities_rejects_unrelated_fields(monkeypatch):
    with mux_profile_scope(monkeypatch, _SCOPE):
        output = json.loads(app_workspace_tool({
            "action": "sync_capabilities", "slug": _SLUG, "expected_instance_id": _INSTANCE,
            "path": "appai.go",
        }))
    assert output["ok"] is False
