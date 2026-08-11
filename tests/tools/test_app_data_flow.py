"""Flow coverage for declared App Data discovery and loopback dispatch."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import pytest

from gateway.session_context import _VAR_MAP
from tests.tools._profile_scope import mux_profile_scope
from tools.app_data_tool import app_data_tool
from tools.registry import discover_builtin_tools, registry
from toolsets import TOOLSETS, _HERMES_CORE_TOOLS, resolve_toolset


_SLUG = "customer-workbench"
_READ_OPERATION = "records.list"
_MUTATION_OPERATION = "records.store"
_CAPABILITY_DIGEST = "a" * 64
_CHANGED_CAPABILITY_DIGEST = "b" * 64


class _AppDataHandler(BaseHTTPRequestHandler):
    calls = []
    rotate_capabilities_before_post = False

    def _send(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.__class__.calls.append(("GET", self.path, self.headers, None))
        assert self.path == f"/api/v1/internal/apps/{_SLUG}/capabilities"
        self._send({
            "version": 1,
            "operations": [
                {"name": _READ_OPERATION, "mode": "read"},
                {"name": _MUTATION_OPERATION, "mode": "mutation"},
            ],
            "capability_digest": _CAPABILITY_DIGEST,
            "owner_user_id": "must-not-project",
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.calls.append(("POST", self.path, self.headers, body))
        expected_digest = (
            _CHANGED_CAPABILITY_DIGEST
            if self.__class__.rotate_capabilities_before_post
            else _CAPABILITY_DIGEST
        )
        if body.get("capability_digest") != expected_digest:
            self._send(
                {
                    "code": "capability_changed",
                    "message": "app data capabilities changed",
                },
                status=409,
            )
            return
        operation = self.path.rsplit("/", 1)[-1]
        self._send({"operation": operation, "received": body})

    def log_message(self, _format, *_args):
        return


@contextmanager
def _app_data_server():
    _AppDataHandler.calls = []
    _AppDataHandler.rotate_capabilities_before_post = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AppDataHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/api/v1/internal/apps", _AppDataHandler.calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _scope(base_url):
    return {
        "ZET_APPHOST_BASE_URL": base_url,
        "ZETTLAB_AGENT_ACTION_TOKEN": "flow-operation-token",
        "ZET_AGENT_ID": "flow-agent",
    }


@pytest.fixture(autouse=True)
def _zet_agent_session_scope():
    platform_var = _VAR_MAP["HERMES_SESSION_PLATFORM"]
    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    platform_token = platform_var.set("zet_agent")
    cron_token = cron_var.set("")
    try:
        yield
    finally:
        cron_var.reset(cron_token)
        platform_var.reset(platform_token)


@contextmanager
def _runtime_scope(platform, cron=""):
    platform_var = _VAR_MAP["HERMES_SESSION_PLATFORM"]
    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    platform_token = platform_var.set(platform)
    cron_token = cron_var.set(cron)
    try:
        yield
    finally:
        cron_var.reset(cron_token)
        platform_var.reset(platform_token)


@contextmanager
def _cron_scope():
    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    token = cron_var.set("1")
    try:
        yield
    finally:
        cron_var.reset(token)


def test_tool_is_discovered_only_on_generated_app_agent_surface(monkeypatch):
    discover_builtin_tools()
    entry = registry.get_entry("app_data")
    assert entry is not None
    assert entry.toolset == "zettlab_apphost"
    assert "app_data" in TOOLSETS["zettlab_apphost"]["tools"]
    assert "app_data" in resolve_toolset("hermes-zet-agent")
    assert "app_data" not in resolve_toolset("hermes-cron")
    assert "app_data" not in _HERMES_CORE_TOOLS
    assert "app_data" not in resolve_toolset("hermes-telegram")
    with mux_profile_scope(
        monkeypatch,
        _scope("http://127.0.0.1:19090/api/v1/internal/apps"),
    ):
        assert entry.check_fn() is True


def test_ordinary_zet_chat_cache_hit_keeps_core_connector_and_app_tools(
    monkeypatch,
):
    """Real assembly keeps neighboring Chat tools when Tool Search is active."""
    import model_tools
    import tools.registry as registry_module
    import tools.tool_search as tool_search_module

    discover_builtin_tools()
    for key, value in {
        "ZET_APPHOST_BASE_URL": (
            "http://127.0.0.1:19090/api/v1/internal/apps"
        ),
        "ZETTLAB_AGENT_ACTION_TOKEN": "flow-operation-token",
        "ZET_AGENT_ID": "flow-agent",
        "ZET_CHAT_APPEND_URL": (
            "http://127.0.0.1:19090/api/v1/internal/chat/append"
        ),
    }.items():
        monkeypatch.setenv(key, value)

    config = tool_search_module.ToolSearchConfig.from_raw({"enabled": "on"})
    monkeypatch.setattr(tool_search_module, "load_config", lambda: config)
    monkeypatch.setattr(model_tools, "_resolve_active_context_length", lambda: 1)

    registry_module.invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    try:
        with _runtime_scope("zet_agent"):
            first = model_tools.get_tool_definitions(
                enabled_toolsets=["hermes-zet-agent", "cronjob"],
                quiet_mode=True,
            )
            second = model_tools.get_tool_definitions(
                enabled_toolsets=["hermes-zet-agent", "cronjob"],
                quiet_mode=True,
            )
        first_names = {item["function"]["name"] for item in first}
        second_names = {item["function"]["name"] for item in second}
        assert first_names == second_names
        assert {
            "todo",
            "list_my_connectors",
            "app_data",
            "tool_search",
            "tool_describe",
            "tool_call",
        }.issubset(first_names)
        assert len(model_tools._tool_defs_cache) == 1
    finally:
        registry_module.invalidate_check_fn_cache()
        model_tools._clear_tool_defs_cache()


@pytest.mark.parametrize(
    ("enabled_toolsets", "skip_tool_search_assembly"),
    [
        (["zettlab_apphost"], False),
        (["zettlab_apphost"], True),
        (["hermes-zet-agent"], False),
        (["hermes-zet-agent"], True),
    ],
)
def test_app_data_schema_cache_isolated_by_platform_and_cron_scope(
    monkeypatch, enabled_toolsets, skip_tool_search_assembly
):
    import model_tools
    import tools.registry as registry_module

    discover_builtin_tools()
    secrets = _scope("http://127.0.0.1:19090/api/v1/internal/apps")

    def secret(name, default=""):
        return secrets.get(name, default)

    def names():
        return {
            definition["function"]["name"]
            for definition in model_tools.get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=skip_tool_search_assembly,
            )
        }

    registry_module.invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    try:
        with patch("agent.secret_scope.is_multiplex_active", return_value=False), patch(
            "tools.app_data_tool._base_url",
            return_value=secrets["ZET_APPHOST_BASE_URL"],
        ), patch("tools.app_data_tool._secret", side_effect=secret):
            with _runtime_scope("zet_agent"):
                assert "app_data" in names()
            with _runtime_scope("zet_agent"), patch(
                "agent.delegation_context.is_delegated_child_context",
                side_effect=RuntimeError("scope unavailable"),
            ):
                assert "app_data" not in names()
            with _runtime_scope("telegram"):
                assert "app_data" not in names()
            with _runtime_scope("zet_agent", cron="1"):
                assert "app_data" not in names()
            with _runtime_scope("zet_agent"):
                assert "app_data" in names()

            model_tools._clear_tool_defs_cache()
            registry_module.invalidate_check_fn_cache()
            with _runtime_scope("zet_agent", cron="1"):
                assert "app_data" not in names()
            with _runtime_scope("zet_agent"):
                assert "app_data" in names()
    finally:
        registry_module.invalidate_check_fn_cache()
        model_tools._clear_tool_defs_cache()


def test_all_tools_and_explicit_messaging_misconfiguration_fail_closed(monkeypatch):
    import model_tools
    import tools.registry as registry_module

    discover_builtin_tools()
    with mux_profile_scope(
        monkeypatch,
        _scope("http://127.0.0.1:19090/api/v1/internal/apps"),
    ):
        for platform, enabled in (
            ("", None),
            ("api_server", ["zettlab_apphost"]),
            ("telegram", ["zettlab_apphost"]),
        ):
            registry_module.invalidate_check_fn_cache()
            model_tools._clear_tool_defs_cache()
            with _runtime_scope(platform):
                names = {
                    definition["function"]["name"]
                    for definition in model_tools.get_tool_definitions(
                        enabled_toolsets=enabled,
                        quiet_mode=True,
                        skip_tool_search_assembly=True,
                    )
                }
            assert "app_data" not in names


def test_delegated_child_cannot_inherit_app_host_or_app_data(monkeypatch):
    import model_tools
    from hermes_cli.tools_config import _get_platform_tools
    from tools.delegate_tool import DELEGATE_BLOCKED_TOOLS, _blocked_toolsets_for_role

    assert {"app_host", "app_data"}.issubset(DELEGATE_BLOCKED_TOOLS)
    config = {
        "platform_toolsets": {
            "zet_agent": ["hermes-zet-agent", "cronjob"],
        },
    }
    enabled = sorted(
        _get_platform_tools(config, "zet_agent", include_default_mcp_servers=False)
    )
    with mux_profile_scope(
        monkeypatch,
        _scope("http://127.0.0.1:19090/api/v1/internal/apps"),
    ):
        parent_names = {
            definition["function"]["name"]
            for definition in model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
            )
        }
        child_names = {
            definition["function"]["name"]
            for definition in model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                disabled_toolsets=_blocked_toolsets_for_role("worker"),
                quiet_mode=True,
            )
        }

    # app_data is the new direct compatibility surface. app_host keeps its
    # existing Tool Search deferral behavior and is covered at dispatch below.
    assert "app_data" in parent_names
    assert {"app_host", "app_data"}.isdisjoint(child_names)


def test_delegated_child_cannot_forge_direct_app_dispatch(monkeypatch):
    discover_builtin_tools()
    for tool_name in ("app_host", "app_data"):
        entry = registry.get_entry(tool_name)
        assert entry is not None
        handler = Mock(return_value=json.dumps({"ok": True}))
        monkeypatch.setattr(entry, "handler", handler)

        with patch(
            "agent.delegation_context.is_delegated_child_context",
            return_value=False,
        ):
            assert json.loads(registry.dispatch(tool_name, {})) == {"ok": True}
        handler.assert_called_once()
        handler.reset_mock()

        with patch(
            "agent.delegation_context.is_delegated_child_context",
            return_value=True,
        ):
            output = json.loads(registry.dispatch(tool_name, {}))

        assert output["error_type"] == "delegated_child_scope"
        assert output["tool"] == tool_name
        handler.assert_not_called()

        with patch(
            "agent.delegation_context.is_delegated_child_context",
            side_effect=RuntimeError("scope unavailable"),
        ):
            lookup_failure = json.loads(registry.dispatch(tool_name, {}))

        assert lookup_failure["error_type"] == "delegated_child_scope"
        assert lookup_failure["tool"] == tool_name
        handler.assert_not_called()


def test_declared_operations_flow_through_real_loopback_transport(monkeypatch):
    with _app_data_server() as (base_url, calls), mux_profile_scope(
        monkeypatch,
        _scope(base_url),
        poison_environ=True,
    ), patch(
        "tools.app_data_tool._approval_result",
        return_value={"approved": True},
    ):
        capability = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))
        read_result = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
            "query": {"label": "product"},
        }))
        mutation_result = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": {"record_id": "record-1", "state": "accepted"},
            "idempotency_key": "record:record-1:1",
        }))

    assert capability["data"] == {
        "version": 1,
        "operations": [
            {"name": _READ_OPERATION, "mode": "read"},
            {"name": _MUTATION_OPERATION, "mode": "mutation"},
        ],
    }
    assert capability["untrusted_app_data"] is True
    assert read_result["data"]["received"] == {
        "capability_digest": _CAPABILITY_DIGEST,
        "query": {"label": "product"},
    }
    assert mutation_result["data"]["received"] == {
        "capability_digest": _CAPABILITY_DIGEST,
        "payload": {"record_id": "record-1", "state": "accepted"},
        "idempotency_key": "record:record-1:1",
    }
    assert [method for method, *_rest in calls] == ["GET", "GET", "POST", "GET", "POST"]
    assert all(
        headers.get("X-Zettlab-Agent-Action-Token") == "flow-operation-token"
        for _method, _path, headers, _body in calls
    )
    assert calls[2][1].endswith(f"/operations/{_READ_OPERATION}")
    assert calls[4][1].endswith(f"/operations/{_MUTATION_OPERATION}")
    assert all("flow-agent" not in json.dumps(body) for *_prefix, body in calls if body)


def test_sensitive_aliases_never_reach_real_loopback_transport(monkeypatch):
    sensitive_fields = (
        "access_key",
        "api_key",
        "apikey",
        "bearer",
        "credential",
        "password",
        "secret",
        "secret_key",
    )
    with _app_data_server() as (base_url, calls), mux_profile_scope(
        monkeypatch,
        _scope(base_url),
        poison_environ=True,
    ):
        for field in sensitive_fields:
            nested_payload = json.loads(app_data_tool({
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "payload": {"outer": [{"nested": {field: "must-not-leak"}}]},
            }))
            query = json.loads(app_data_tool({
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "query": {field: "must-not-leak"},
            }))

            assert nested_payload["error"]["code"] == "invalid_request"
            assert query["error"]["code"] == "invalid_request"

    assert calls == []


def test_capability_change_blocks_read_to_mutation_toctou(monkeypatch):
    with _app_data_server() as (base_url, calls), mux_profile_scope(
        monkeypatch,
        _scope(base_url),
        poison_environ=True,
    ):
        _AppDataHandler.rotate_capabilities_before_post = True
        result = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
        }))

    assert result == {
        "ok": False,
        "error": {
            "code": "capability_changed",
            "message": "app data capabilities changed",
        },
        "status": 409,
    }
    assert [method for method, *_rest in calls] == ["GET", "POST"]
    assert calls[-1][3]["capability_digest"] == _CAPABILITY_DIGEST


def test_cron_real_transport_is_fully_denied(monkeypatch):
    with _app_data_server() as (base_url, calls), mux_profile_scope(
        monkeypatch,
        _scope(base_url),
        poison_environ=True,
    ), _cron_scope(), patch("tools.app_data_tool._approval_result") as approval:
        capabilities = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))
        read_result = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
        }))
        mutation_result = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": {"record_id": "record-1"},
            "idempotency_key": "record:record-1",
        }))

    assert capabilities["error"]["code"] == "cron_scope_denied"
    assert read_result["error"]["code"] == "cron_scope_denied"
    assert mutation_result["error"]["code"] == "cron_scope_denied"
    assert calls == []
    approval.assert_not_called()
