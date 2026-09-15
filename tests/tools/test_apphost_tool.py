"""Tests for tools/apphost_tool.py.

The load-bearing behavior is shared-gateway readiness: every credential the
tool needs resolves through the profile secret scope, so it must keep working
when ``os.environ`` holds none of the values (or holds another profile's stale
ones). Scope handling is asserted by iterating the scope mapping — not by
naming individual variables — so a change to the secret set stays red until
the tests are updated.
"""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from tests.tools._profile_scope import mux_profile_scope, request_fingerprint
from tools.apphost_tool import (
    APP_HOST_SCHEMA,
    _CALL_HTTP_METHODS,
    _check_app_host,
    _execution_headers,
    _local_error,
    app_host_tool,
)

_BASE_URL = "http://127.0.0.1:18080/api/v1/internal/apphost"


def _scope(**extra):
    scope = {
        "ZET_APPHOST_BASE_URL": _BASE_URL,
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-apphost-token",
    }
    scope.update(extra)
    return scope


class _Headers:
    """Raw-header semantics: get("Content-Type") is None when the server sent
    none (a real 204), unlike email.Message.get_content_type() which would
    default to text/plain."""

    def __init__(self, content_type):
        self._content_type = content_type

    def get(self, name, default=None):
        if name.lower() == "content-type" and self._content_type is not None:
            return self._content_type
        return default


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status
        self.headers = _Headers("application/json")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps(self._payload).encode("utf-8")


class _RawResp:
    """A 2xx response with a verbatim byte body and content type — the shape
    the upstream really uses for 204 No Content (no Content-Type header at
    all) and text/plain logs."""

    def __init__(self, status, body=b"", content_type=None):
        self.status = status
        self._body = body
        self.headers = _Headers(content_type)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self._body


def _capture_urlopen(seen, payload=None, status=200):
    def _open(req, timeout=None):
        seen["req"] = req
        seen["timeout"] = timeout
        return _Resp(payload if payload is not None else {"code": 200, "data": {}}, status=status)

    return _open


# --- availability gate -------------------------------------------------------

def test_check_gate_requires_every_secret(monkeypatch):
    full = _scope()
    with mux_profile_scope(monkeypatch, full):
        assert _check_app_host() is True
    # Removing ANY of the gating secrets must close the gate — iterate the
    # scope so a new required secret cannot be forgotten silently.
    for missing in full:
        partial = {k: v for k, v in full.items() if k != missing}
        with mux_profile_scope(monkeypatch, {**partial, missing: ""}):
            assert _check_app_host() is False, f"gate stayed open without {missing}"


def test_check_gate_rejects_malformed_base_url(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope(ZET_APPHOST_BASE_URL="not-a-url")):
        assert _check_app_host() is False


def test_app_data_url_never_substitutes_for_app_host_url(monkeypatch):
    with mux_profile_scope(
        monkeypatch,
        _scope(
            ZET_APPHOST_BASE_URL="",
            ZET_APP_DATA_BASE_URL=(
                "http://127.0.0.1:18080/api/v1/internal/apps"
            ),
        ),
    ):
        assert _check_app_host() is False


def test_check_gate_is_profile_scope_sensitive():
    # The registry must re-evaluate this gate per profile scope instead of
    # serving a TTL-cached verdict across multiplexed profiles.
    assert getattr(_check_app_host, "_profile_scope_sensitive") is True


def test_registered_with_gate():
    from tools.registry import registry

    entry = registry.get_entry("app_host")
    assert entry is not None
    assert entry.check_fn is _check_app_host


# --- shared-gateway scope flow ----------------------------------------------

def test_profile_scope_flow_works_with_poisoned_environ(monkeypatch):
    """Scope has the values, os.environ holds another profile's stale decoys:
    the request must be built entirely from the scope."""
    scope = _scope()
    seen = {}
    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, {"apps": []})):
            out = app_host_tool({"action": "list"})
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["ok"] is True and parsed["data"] == {"apps": []}
    req = seen["req"]
    assert req.full_url == _BASE_URL + "?mine=1"
    assert req.get_header("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)


def test_profile_scope_flow_works_with_empty_environ(monkeypatch):
    scope = _scope()
    seen = {}
    with mux_profile_scope(monkeypatch, scope):  # scope keys purged from env
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is True
    assert seen["req"].full_url == _BASE_URL + "/storage"


def test_request_forwards_only_task_local_execution_headers(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        clear_turn_vars,
        pop_zettlab_auth_principal,
        push_zettlab_auth_principal,
        set_session_vars,
        set_turn_vars,
    )
    seen = {}
    session_tokens = set_session_vars(
        session_id="cron_task_abcdef123456_20260817_120000",
        session_key="zettlab:owner-1:agent-1:stable-session",
    )
    turn_tokens = set_turn_vars(
        turn_id="turn-1", hardware_execution_token="a" * 64
    )
    principal_token = push_zettlab_auth_principal("iam:issuer:user:owner-1")
    try:
        with mux_profile_scope(monkeypatch, _scope()), patch(
            "tools.apphost_tool._urlopen", _capture_urlopen(seen)
        ):
            assert json.loads(app_host_tool({"action": "probe"}))["ok"] is True
    finally:
        pop_zettlab_auth_principal(principal_token)
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
    req = seen["req"]
    retired_header = "X-zettlab-business-" + "execution-token"
    assert req.get_header(retired_header) is None
    assert req.get_header("X-zettlab-hardware-execution-token") is None
    assert req.get_header("X-zettlab-auth-principal-id") == (
        "iam:issuer:user:owner-1"
    )
    assert req.get_header("X-hermes-turn-id") == "turn-1"
    assert req.get_header("X-hermes-session-id") == "cron_task_abcdef123456_20260817_120000"
    assert req.get_header("X-hermes-session-key") == (
        "zettlab:owner-1:agent-1:stable-session"
    )
    assert req.get_header("X-zettlab-app-maintenance-task-id") == "abcdef123456"


def test_hardware_execution_token_is_never_forwarded_by_apphost(monkeypatch):
    from gateway.session_context import clear_turn_vars, set_turn_vars

    seen = {}
    turn_tokens = set_turn_vars(hardware_execution_token="b" * 64)
    try:
        with mux_profile_scope(monkeypatch, _scope()), patch(
            "tools.apphost_tool._urlopen", _capture_urlopen(seen)
        ):
            assert json.loads(app_host_tool({"action": "probe"}))["ok"] is True
    finally:
        clear_turn_vars(turn_tokens)
    retired_header = "X-zettlab-business-" + "execution-token"
    assert seen["req"].get_header(retired_header) is None
    assert seen["req"].get_header("X-zettlab-hardware-execution-token") is None
    assert seen["req"].get_header("X-hermes-turn-id") is None


@pytest.mark.parametrize(
    "field,value,missing_header",
    [
        ("principal", "iam:issuer:user:owner\r\nforged", "X-Zettlab-Auth-Principal-Id"),
        ("principal", "p" * 513, "X-Zettlab-Auth-Principal-Id"),
        ("turn_id", "turn\x00forged", "X-Hermes-Turn-Id"),
        ("turn_id", "t" * 257, "X-Hermes-Turn-Id"),
        ("session_id", " session-1", "X-Hermes-Session-Id"),
        ("session_id", "s" * 257, "X-Hermes-Session-Id"),
        ("session_key", "zettlab:owner:agent:key\nforged", "X-Hermes-Session-Key"),
        ("session_key", "k" * 257, "X-Hermes-Session-Key"),
    ],
)
def test_execution_headers_drop_invalid_or_oversized_values(
    field, value, missing_header
):
    values = {
        "principal": "iam:issuer:user:owner-1",
        "turn_id": "turn-1",
        "session_id": "session-1",
        "session_key": "zettlab:owner-1:agent-1:session-1",
    }
    values[field] = value

    def session_env(name, default=""):
        return {
            "HERMES_SESSION_ID": values["session_id"],
            "HERMES_SESSION_KEY": values["session_key"],
        }.get(name, default)

    with patch(
        "gateway.session_context.zettlab_auth_principal",
        return_value=values["principal"],
    ), patch(
        "gateway.session_context.current_turn_identity",
        return_value=(values["turn_id"], object()),
    ), patch(
        "gateway.session_context.get_session_env",
        side_effect=session_env,
    ):
        headers = _execution_headers()

    assert missing_header not in headers
    assert all("\r" not in item and "\n" not in item and "\x00" not in item for item in headers.values())


def test_app_host_request_keeps_its_own_base_url(monkeypatch):
    seen = {}
    with mux_profile_scope(
        monkeypatch,
        _scope(
            ZET_APP_DATA_BASE_URL=(
                "http://127.0.0.1:18080/api/v1/internal/apps"
            )
        ),
    ), patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
        out = json.loads(app_host_tool({"action": "probe"}))

    assert out["ok"] is True
    assert seen["req"].full_url == _BASE_URL + "/storage"


@pytest.mark.parametrize("action,args,method,path,body", [
    ("probe", {}, "GET", "/storage", None),
    ("list", {}, "GET", "?mine=1", None),
    ("acquire_slot", {}, "POST", "/buildslot", None),
    ("release_slot", {"slot_token": "s1"}, "DELETE", "/buildslot/s1", None),
    ("publish", {"mode": "install", "source_subdir": "runs/run-1/app1",
                 "data_refresh": "static"},
     "POST", "/publish",
     {"mode": "install", "source_subdir": "runs/run-1/app1",
      "data_refresh": "static"}),
    ("publish", {"mode": "reload", "source_subdir": "runs/run-2/app1",
                 "note": "Footer 加了一个链接"},
     "POST", "/publish",
     {"mode": "reload", "source_subdir": "runs/run-2/app1",
      "note": "Footer 加了一个链接"}),
    ("install", {"staging_dir": "/tmp/stage", "slug": "app1",
                 "data_refresh": "static"}, "POST", "/install",
     {"staging_dir": "/tmp/stage", "slug": "app1", "data_refresh": "static"}),
    ("reload", {"slug": "app1", "staging_dir": "/tmp/stage"}, "POST", "/app1/reload",
     {"staging_dir": "/tmp/stage"}),
    # The note travels with the version and is what the user is shown when
    # deciding whether to undo it, so it has to reach the host.
    ("reload", {"slug": "app1", "staging_dir": "/tmp/stage", "note": "Footer 加了一个链接"},
     "POST", "/app1/reload", {"staging_dir": "/tmp/stage", "note": "Footer 加了一个链接"}),
    # Undo always names the version it means. This is what makes a retry
    # safe — an undo that already succeeded is recognised instead of swapping
    # the app forward again. The untargeted form the server would accept is
    # deliberately not offered on this face (see the required-params test).
    ("rollback", {"slug": "app1", "to_version": "v17858"}, "POST", "/app1/rollback",
     {"to_version": "v17858"}),
    ("delete", {"slug": "app1"}, "DELETE", "/app1", None),
    ("lifecycle", {"slug": "app1", "lifecycle_action": "restart"}, "POST",
     "/app1/lifecycle", {"action": "restart"}),
    ("logs", {"slug": "app1", "tail": 50}, "GET", "/app1/logs?tail=50", None),
    ("app_capabilities", {"slug": "app1"}, "GET", "/app1/capabilities", None),
    ("app_operation", {"slug": "app1", "app_operation": "summary", "payload": {"range": "week"}, "capability_digest": "a" * 64}, "POST", "/app1/operations/summary", {"payload": {"range": "week"}, "capability_digest": "a" * 64}),
    # call rides POST /{slug}/call with method/path/body in the request body:
    # the app path is payload, never URL — the server builds the target URL
    # from the slug (the agent has no host/port to give).
    ("call", {"slug": "app1", "path": "/api/refresh", "http_method": "POST",
              "body": {"source": "cron"}},
     "POST", "/app1/call",
     {"method": "POST", "path": "/api/refresh", "body": {"source": "cron"}}),
])
def test_action_routing_flow(monkeypatch, action, args, method, path, body):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope(), poison_environ=True):
        completion_status = 204 if action in {"release_slot", "delete"} else 200
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, status=completion_status)):
            out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is True
    req = seen["req"]
    assert req.get_method() == method
    assert req.full_url == _BASE_URL + path
    if body is None:
        assert req.data is None
    else:
        assert json.loads(req.data.decode("utf-8")) == body
    assert "stale-" not in request_fingerprint(req)


# --- string contract ---------------------------------------------------------

def test_handler_always_returns_json_string(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen({})):
            success = app_host_tool({"action": "list"})
    unknown = app_host_tool({"action": "definitely_not_an_action"})
    missing = app_host_tool({"action": "release_slot"})
    for out in (success, unknown, missing):
        assert isinstance(out, str) and not isinstance(out, dict)
        json.loads(out)  # must be valid JSON


# --- completion status contract ----------------------------------------------

# Every HTTP action with the minimal args to reach the network layer.
@pytest.mark.parametrize("action,args", [
    ("release_slot", {"slot_token": "s1"}),
    ("delete", {"slug": "app1"}),
])
def test_204_is_success_only_for_actions_with_a_204_completion_contract(monkeypatch, action, args):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=_RawResp(204)):
            out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is True
    assert out["data"] == {}


# Typed capability and workflow-journal routes have dedicated wire-shape and
# completion-contract coverage above; this table keeps the ordinary routes
# compact without weakening those stricter assertions.
_ALL_HTTP_ACTION_ARGS = [
    ("probe", {}), ("list", {}), ("acquire_slot", {}),
    ("release_slot", {"slot_token": "s1"}),
    ("publish", {"mode": "install", "source_subdir": "runs/run-1/app1", "data_refresh": "static"}),
    ("install", {"staging_dir": "/tmp/s", "slug": "app1", "data_refresh": "static"}),
    ("reload", {"slug": "app1", "staging_dir": "/tmp/s"}),
    ("rollback", {"slug": "app1", "to_version": "v1"}),
    ("delete", {"slug": "app1"}),
    ("lifecycle", {"slug": "app1", "lifecycle_action": "restart"}),
    ("logs", {"slug": "app1"}),
    ("call", {"slug": "app1", "path": "/api/refresh", "http_method": "POST"}),
]


@pytest.mark.parametrize("status", [201, 202, 204])
def test_non_completion_2xx_is_outcome_unknown(monkeypatch, status):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=_RawResp(status)):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is False
    assert out["error"]["code"] == "outcome_unknown"
    assert out["status"] == status


def test_app_operation_requires_capability_digest_before_sending(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch("tools.apphost_tool._urlopen") as open_request:
        out = json.loads(app_host_tool({
            "action": "app_operation", "slug": "app1", "app_operation": "summary", "payload": {},
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"
    assert out["status"] == 0
    open_request.assert_not_called()


# --- ADIC v1: turn-scoped data.import ledger ---------------------------------
# app_host records EVERY app_operation outcome, tagged with its operation
# name, into a bounded, turn-scoped ledger (gateway.session_context) that
# cron/scheduler.py reads right before mark_job_run to judge success by
# whether the app's own declared write operation actually landed this round,
# not by whether the agent produced a plausible reply. See
# zettlab-local-docs/app-fullstack/2026-08-17-应用数据导入契约-ADIC-v1.md §4.5
# and the paired interface-freeze doc §7-8. Recording is deliberately NOT
# filtered to the literal "data.import" here — local-server stamps
# job["import_operation"] with the app's own declared mutation name (e.g.
# "records.refresh" for a blueprint app), and cron/scheduler.py does the name
# filtering at verdict time against that per-job value. A read call like
# data.import_schema IS recorded (see the test below) — it is excluded from
# the verdict purely because its name never matches any job's
# import_operation, not because this layer special-cases read calls. The
# call() two-layer status (tested above) is a completely separate code path
# and must stay untouched.

_IMPORT_ARGS = {
    "action": "app_operation", "slug": "hangzhou-weather-live",
    "app_operation": "data.import",
    "payload": {"daily": [{"forecast_date": "2026-08-18"}]},
    "capability_digest": "b" * 64,
}


def test_data_import_success_is_recorded_in_active_ledger(monkeypatch):
    from gateway.session_context import (
        import_attempts_snapshot, pop_import_attempts_scope, push_import_attempts_scope,
    )
    token = push_import_attempts_scope()
    try:
        with mux_profile_scope(monkeypatch, _scope()):
            with patch(
                "tools.apphost_tool._urlopen",
                _capture_urlopen({}, {"import_receipt": {"committed": True}}),
            ):
                out = json.loads(app_host_tool(_IMPORT_ARGS))
        assert out["ok"] is True
        ledger = import_attempts_snapshot()
    finally:
        pop_import_attempts_scope(token)
    assert ledger == [{
        "operation": "data.import", "ok": True, "error_code": "", "error_message": "",
    }]


def test_data_import_rejection_is_recorded_with_upstream_code(monkeypatch):
    upstream = {"code": "import_rejected", "message": "湿度必须是 0-100 的整数"}
    from gateway.session_context import (
        import_attempts_snapshot, pop_import_attempts_scope, push_import_attempts_scope,
    )
    token = push_import_attempts_scope()
    try:
        with mux_profile_scope(monkeypatch, _scope()):
            with patch(
                "tools.apphost_tool._urlopen",
                _http_error(400, json.dumps(upstream).encode("utf-8")),
            ):
                out = json.loads(app_host_tool(_IMPORT_ARGS))
        assert out["ok"] is False and out["error"]["code"] == "import_rejected"
        ledger = import_attempts_snapshot()
    finally:
        pop_import_attempts_scope(token)
    assert ledger == [{
        "operation": "data.import", "ok": False, "error_code": "import_rejected",
        "error_message": "湿度必须是 0-100 的整数",
    }]


def test_data_import_not_confirmed_is_recorded_with_upstream_code(monkeypatch):
    """502 import_not_confirmed (2xx from the app but no valid receipt) must
    be distinguishable from import_rejected in the ledger, per the interface
    freeze's error-code table (§4)."""
    upstream = {"code": "import_not_confirmed", "message": "app answered without a receipt"}
    from gateway.session_context import (
        import_attempts_snapshot, pop_import_attempts_scope, push_import_attempts_scope,
    )
    token = push_import_attempts_scope()
    try:
        with mux_profile_scope(monkeypatch, _scope()):
            with patch(
                "tools.apphost_tool._urlopen",
                _http_error(502, json.dumps(upstream).encode("utf-8")),
            ):
                out = json.loads(app_host_tool(_IMPORT_ARGS))
        assert out["ok"] is False and out["error"]["code"] == "import_not_confirmed"
        ledger = import_attempts_snapshot()
    finally:
        pop_import_attempts_scope(token)
    assert ledger[0]["error_code"] == "import_not_confirmed"


def test_data_import_schema_read_is_recorded_under_its_own_operation_name(monkeypatch):
    """A read call (data.import_schema) IS recorded — this layer does not
    special-case reads. It is kept out of a job's import verdict purely
    because cron/scheduler.py filters the ledger by job["import_operation"],
    and "data.import_schema" never equals that value. If this layer instead
    pre-filtered by name, an app whose declared write operation isn't
    literally "data.import" (e.g. "records.refresh") would never get
    anything recorded and would fail every round — see the P0 this test
    guards against in tests/cron/test_import_contract_verdict.py."""
    from gateway.session_context import (
        import_attempts_snapshot, pop_import_attempts_scope, push_import_attempts_scope,
    )
    token = push_import_attempts_scope()
    try:
        with mux_profile_scope(monkeypatch, _scope()):
            with patch("tools.apphost_tool._urlopen", _capture_urlopen({}, {"daily_forecast": {}})):
                app_host_tool({
                    "action": "app_operation", "slug": "app1",
                    "app_operation": "data.import_schema", "payload": {},
                    "capability_digest": "c" * 64,
                })
        ledger = import_attempts_snapshot()
    finally:
        pop_import_attempts_scope(token)
    assert ledger == [{
        "operation": "data.import_schema", "ok": True, "error_code": "", "error_message": "",
    }]


def test_call_action_never_touches_the_import_ledger(monkeypatch):
    """The legacy call() two-layer status (app-level 400/500 arrives as
    ok:true) must never be mistaken for a data.import outcome."""
    from gateway.session_context import (
        import_attempts_snapshot, pop_import_attempts_scope, push_import_attempts_scope,
    )
    token = push_import_attempts_scope()
    try:
        payload = {"status": 500, "content_type": "application/json", "body": {"error": "boom"}}
        with mux_profile_scope(monkeypatch, _scope()):
            with patch("tools.apphost_tool._urlopen", _capture_urlopen({}, payload)):
                out = json.loads(app_host_tool(dict(_CALL_ARGS)))
        assert out["ok"] is True
        ledger = import_attempts_snapshot()
    finally:
        pop_import_attempts_scope(token)
    assert ledger == []


def test_data_import_outside_a_pushed_scope_is_a_silent_noop(monkeypatch):
    """Interactive turns never push a ledger scope. Recording must not raise
    and must not fabricate a ledger visible to a later reader."""
    from gateway.session_context import import_attempts_snapshot
    with mux_profile_scope(monkeypatch, _scope()):
        with patch(
            "tools.apphost_tool._urlopen",
            _capture_urlopen({}, {"import_receipt": {"committed": True}}),
        ):
            out = json.loads(app_host_tool(_IMPORT_ARGS))
    assert out["ok"] is True
    assert import_attempts_snapshot() == []


def test_publish_operation_is_passed_through_unchanged(monkeypatch):
    from gateway.session_context import (
        clear_session_vars, clear_turn_vars, pop_zettlab_auth_principal,
        push_zettlab_auth_principal, set_session_vars, set_turn_vars,
    )

    seen = {}
    operation = {"operation_id": "op-1", "purpose": "每天同步汇率", "data_refresh": "user_confirmed_auto", "maintenance": {"schedule": "0 9 * * *"}}
    session_tokens = set_session_vars(session_id="session-1")
    turn_tokens = set_turn_vars(turn_id="turn-1")
    principal_token = push_zettlab_auth_principal("iam:user-1")
    try:
        with mux_profile_scope(monkeypatch, _scope(ZET_AGENT_ID="main")), patch(
            "tools.apphost_tool.request_app_auto_refresh_token", return_value="a" * 64
        ) as mint, patch(
            "tools.apphost_tool._urlopen",
            _capture_urlopen(seen, {"operation": {"operation_id": "op-1", "terminal": "succeeded"}}),
        ):
            out = json.loads(app_host_tool({
                "action": "publish", "mode": "install", "source_subdir": "runs/app",
                "data_refresh": "user_confirmed_auto", "operation": operation,
            }))
    finally:
        pop_zettlab_auth_principal(principal_token)
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
    assert out["ok"] is True
    mint.assert_called_once()
    assert mint.call_args.kwargs["owner_agent_id"] == "main"
    assert mint.call_args.kwargs["turn_id"] == "turn-1"
    assert mint.call_args.kwargs["session_id"] == "session-1"
    assert mint.call_args.kwargs["operation_kind"] == "apphost_publish_v1"
    assert mint.call_args.kwargs["operation"] == {
        "mode": "install",
        "source_subdir": "runs/app",
        "data_refresh": "user_confirmed_auto",
        "operation": operation,
    }
    assert mint.call_args.kwargs["owner_principal"] == "iam:user-1"
    assert json.loads(seen["req"].data)["operation"] == operation
    assert seen["req"].get_header("X-zettlab-agent-action-token") == "a" * 64


def test_auto_publish_requires_an_active_user_turn_before_minting_scope(monkeypatch):
    operation = {"operation_id": "op-1", "data_refresh": "user_confirmed_auto"}
    with mux_profile_scope(monkeypatch, _scope(ZET_AGENT_ID="main")), patch(
        "tools.apphost_tool.request_app_auto_refresh_token"
    ) as mint, patch("tools.apphost_tool._urlopen") as open_request:
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "user_confirmed_auto", "operation": operation,
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "automatic_maintenance_unavailable"
    assert out["status"] == 0
    mint.assert_not_called()
    open_request.assert_not_called()


def test_auto_publish_without_operation_is_rejected_before_credentials_or_network(monkeypatch):
    with patch("tools.apphost_tool._secret", side_effect=AssertionError("secret must not be read")), patch(
        "tools.apphost_tool.request_app_auto_refresh_token", side_effect=AssertionError("scope must not be minted")
    ), patch("tools.apphost_tool._urlopen", side_effect=AssertionError("network must not be used")):
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "user_confirmed_auto",
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"
    assert out["status"] == 0


def test_operation_enabled_publish_202_returns_verified_pending_receipt(monkeypatch):
    operation = {"operation_id": "op-1", "purpose": "每天同步汇率", "data_refresh": "static"}
    response = {"operation": {"operation_id": "op-1", "terminal": "pending"}}
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp(response, status=202)
    ):
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "static", "operation": operation,
        }))
    assert out["ok"] is True
    assert out["data"]["outcome"] == "pending"
    assert out["data"]["operation_id"] == "op-1"


def test_operation_enabled_publish_200_returns_verified_terminal_receipt(monkeypatch):
    operation = {"operation_id": "op-1", "purpose": "每天同步汇率", "data_refresh": "static"}
    response = {"operation": {"operation_id": "op-1", "terminal": "succeeded"}}
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp(response, status=200)
    ):
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "static", "operation": operation,
        }))
    assert out["ok"] is True
    assert out["data"]["outcome"] == "completed"
    assert out["data"]["state"] == "succeeded"


def test_operation_enabled_publish_reload_derives_outer_data_refresh_from_intent(monkeypatch):
    from gateway.session_context import (
        clear_session_vars, clear_turn_vars, pop_zettlab_auth_principal,
        push_zettlab_auth_principal, set_session_vars, set_turn_vars,
    )

    seen = {}
    operation = {"operation_id": "op-reload", "data_refresh": "user_confirmed_auto"}
    response = {"operation": {"operation_id": "op-reload", "terminal": "succeeded"}}
    session_tokens = set_session_vars(session_id="session-1")
    turn_tokens = set_turn_vars(turn_id="turn-1")
    principal_token = push_zettlab_auth_principal("iam:user-1")
    try:
        with mux_profile_scope(monkeypatch, _scope(ZET_AGENT_ID="main")), patch(
            "tools.apphost_tool.request_app_auto_refresh_token", return_value="a" * 64
        ), patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, response)):
            out = json.loads(app_host_tool({
                "action": "publish", "mode": "reload", "source_subdir": "runs/app",
                "operation": operation,
            }))
    finally:
        pop_zettlab_auth_principal(principal_token)
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
    assert out["ok"] is True
    assert json.loads(seen["req"].data) == {
        "mode": "reload", "source_subdir": "runs/app",
        "data_refresh": "user_confirmed_auto", "operation": operation,
    }


def test_operation_enabled_publish_rejects_conflicting_outer_data_refresh(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch("tools.apphost_tool._urlopen") as open_request:
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "static",
            "operation": {"operation_id": "op-1", "data_refresh": "user_confirmed_auto"},
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"
    assert out["status"] == 0
    open_request.assert_not_called()


@pytest.mark.parametrize(("action", "args", "hint"), [
    ("install", {
        "slug": "app1", "staging_dir": "/tmp/stage", "data_refresh": "static",
        "operation": {"operation_id": "op-1", "data_refresh": "static"},
    }, "publish(mode=install)"),
    ("reload", {
        "slug": "app1", "staging_dir": "/tmp/stage",
        "operation": {"operation_id": "op-1", "data_refresh": "static"},
    }, "publish(mode=reload)"),
])
def test_legacy_mutations_reject_workflow_operation_before_secret_or_network(monkeypatch, action, args, hint):
    # The local rejection must happen in _build_request before credentials are
    # resolved: an old route cannot accidentally receive or discard a journal
    # intent merely because this profile happens to have a valid token.
    with patch("tools.apphost_tool._secret", side_effect=AssertionError("secret must not be read")), patch(
        "tools.apphost_tool._urlopen", side_effect=AssertionError("network must not be used")
    ):
        out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"
    assert out["status"] == 0
    assert hint in out["error"]["message"]


def test_legacy_install_rejects_auto_refresh_before_secret_or_network(monkeypatch):
    with patch("tools.apphost_tool._secret", side_effect=AssertionError("secret must not be read")), patch(
        "tools.apphost_tool._urlopen", side_effect=AssertionError("network must not be used")
    ):
        out = json.loads(app_host_tool({
            "action": "install", "slug": "weather", "staging_dir": "/tmp/stage",
            "data_refresh": "user_confirmed_auto",
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"
    assert out["status"] == 0
    assert "publish(mode=install)" in out["error"]["message"]


@pytest.mark.parametrize("response", [
    {}, {"operation": {}},
    {"operation": {"operation_id": "op-other", "terminal": "pending"}},
    {"operation": {"operation_id": "op-1", "terminal": "succeeded"}},
])
def test_operation_enabled_publish_202_without_matching_open_receipt_is_unknown(monkeypatch, response):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp(response, status=202)
    ):
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "static", "operation": {"operation_id": "op-1", "data_refresh": "static"},
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "outcome_unknown"
    assert out["operation_id"] == "op-1"


@pytest.mark.parametrize("response", [
    {}, {"operation": {}},
    {"operation": {"operation_id": "op-other", "terminal": "succeeded"}},
    {"operation": {"operation_id": "op-1", "terminal": "pending"}},
])
def test_operation_enabled_publish_200_without_matching_terminal_receipt_is_unknown(monkeypatch, response):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp(response, status=200)
    ):
        out = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/app",
            "data_refresh": "static", "operation": {"operation_id": "op-1", "data_refresh": "static"},
        }))
    assert out["ok"] is False
    assert out["error"]["code"] == "outcome_unknown"
    assert out["operation_id"] == "op-1"


def test_workflow_operation_status_returns_pending_only_from_202_pending(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp({"operation_id": "op-1", "terminal": "pending"}, status=202)
    ):
        out = json.loads(app_host_tool({"action": "workflow_operation_status", "slug": "app1", "operation_id": "op-1"}))
    assert out["ok"] is True
    assert out["data"]["outcome"] == "pending"


def test_workflow_operation_resume_uses_only_the_journal_receipt(monkeypatch):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen",
        _capture_urlopen(seen, {"operation_id": "op-1", "terminal": "succeeded"}),
    ):
        out = json.loads(app_host_tool({"action": "workflow_operation_resume", "slug": "app1", "operation_id": "op-1"}))
    assert out["ok"] is True
    assert seen["req"].full_url == _BASE_URL + "/app1/operation/op-1/resume"
    assert seen["req"].data is None


@pytest.mark.parametrize(("action", "status", "terminal"), [
    ("workflow_operation_status", 202, "pending"),
    ("workflow_operation_resume", 200, "succeeded"),
])
def test_workflow_operation_receipt_must_match_requested_operation_id(monkeypatch, action, status, terminal):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen",
        return_value=_Resp({"operation_id": "op-other", "terminal": terminal}, status=status),
    ):
        out = json.loads(app_host_tool({"action": action, "slug": "app1", "operation_id": "op-1"}))
    assert out["ok"] is False
    assert out["error"]["code"] == "outcome_unknown"
    assert out["operation_id"] == "op-1"


@pytest.mark.parametrize("status,payload", [
    (200, None), (200, {}), (200, {"operation_id": "op-1"}),
    (200, {"operation_id": "op-1", "terminal": "pending"}),
    (202, {"operation_id": "op-1", "terminal": "succeeded"}),
])
def test_workflow_status_without_valid_status_receipt_is_outcome_unknown(monkeypatch, status, payload):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.apphost_tool._urlopen", return_value=_Resp(payload, status=status)
    ):
        out = json.loads(app_host_tool({"action": "workflow_operation_status", "slug": "app1", "operation_id": "op-1"}))
    assert out["ok"] is False
    assert out["error"]["code"] == "outcome_unknown"


def test_2xx_text_plain_body_is_success_with_text_payload(monkeypatch):
    """logs answers 200 text/plain (internal.go GetLogs) — the text IS the
    payload and must come back, not be flattened to an empty object."""
    log_text = "2026-07-30 01:00:00 INFO app started\n2026-07-30 01:00:01 INFO ready\n"
    resp = _RawResp(200, log_text.encode("utf-8"), content_type="text/plain")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "logs", "slug": "app1"}))
    assert out["ok"] is True
    assert out["data"] == {"text": log_text, "truncated": False,
                           "total_chars": len(log_text)}


def test_2xx_declared_json_but_unparseable_is_still_success(monkeypatch):
    resp = _RawResp(200, b"{not json", content_type="application/json")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is True
    assert out["data"]["text"] == "{not json"
    assert out["data"]["truncated"] is False


def test_empty_text_body_keeps_full_text_payload_shape(monkeypatch):
    """A fresh app's log is empty: 200 + text/plain + empty body. The shape
    follows the DECLARED type, not the accident of emptiness — the skill
    reads text/truncated/total_chars without existence checks, so all three
    must be present with total_chars 0 (found live: empty logs answered {}
    and the fields the skill was promised vanished)."""
    resp = _RawResp(200, b"", content_type="text/plain; charset=utf-8")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "logs", "slug": "app1"}))
    assert out["ok"] is True
    assert out["data"] == {"text": "", "truncated": False, "total_chars": 0}


def test_content_free_204_still_answers_empty_object(monkeypatch):
    # A real 204 (release_slot/delete) carries no Content-Type at all —
    # genuinely content-free stays {}, distinct from an empty text body.
    resp = _RawResp(204, b"", content_type=None)
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "release_slot", "slot_token": "s1"}))
    assert out["ok"] is True
    assert out["data"] == {}


def test_oversized_text_payload_is_tail_truncated_and_labeled(monkeypatch):
    """Delivery-layer cap: the transport cap keeps a long-log fetch from
    failing, but the text handed to the model is tail-truncated (errors are
    usually at the end) and labeled so the model knows it saw a partial
    window (and how big the original was)."""
    from tools.apphost_tool import _MAX_TEXT_PAYLOAD_CHARS

    head = "EARLY " * 4000
    tail = "TAIL-MARKER " * 8000
    log_text = head + tail
    assert len(log_text) > _MAX_TEXT_PAYLOAD_CHARS
    resp = _RawResp(200, log_text.encode("utf-8"), content_type="text/plain")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "logs", "slug": "app1"}))
    assert out["ok"] is True
    data = out["data"]
    assert data["truncated"] is True
    assert data["total_chars"] == len(log_text)
    assert len(data["text"]) == _MAX_TEXT_PAYLOAD_CHARS
    assert data["text"] == log_text[-_MAX_TEXT_PAYLOAD_CHARS:]  # the TAIL
    assert data["text"].endswith("TAIL-MARKER ")


# --- upstream error-body pass-through & leak guarantees ----------------------

def _assert_no_secret_leak(out, scope):
    for value in scope.values():
        assert value not in out, f"secret value leaked into tool output: {out}"
    assert "127.0.0.1:18080" not in out  # netloc of the base URL


def test_http_error_passes_upstream_error_body_verbatim(monkeypatch):
    """Skills branch on the upstream `code` string (slug_conflict /
    storage_full / ...), so the {code, message} body must arrive untouched —
    not flattened into prose."""
    scope = _scope()
    upstream_body = {"code": "storage_full", "message": "app data volume has 12MiB free"}

    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 507, "Insufficient Storage", None,
            io.BytesIO(json.dumps(upstream_body).encode("utf-8")),
        )

    with mux_profile_scope(monkeypatch, scope):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = app_host_tool({"action": "install", "slug": "a1",
                                 "staging_dir": "/tmp/s", "data_refresh": "static"})
    parsed = json.loads(out)
    assert parsed["ok"] is False and parsed["status"] == 507
    assert parsed["error"] == upstream_body  # verbatim, key for key
    _assert_no_secret_leak(out, scope)


def test_http_error_without_json_body_degrades_to_transport_error(monkeypatch):
    scope = _scope()

    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 502, "Bad Gateway", None, io.BytesIO(b"<html>bad gateway</html>")
        )

    with mux_profile_scope(monkeypatch, scope):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = app_host_tool({"action": "probe"})
    parsed = json.loads(out)
    assert parsed["ok"] is False and parsed["status"] == 502
    assert parsed["error"]["code"] == "transport_error"  # same shape as upstream
    _assert_no_secret_leak(out, scope)


# --- rollback against a server that predates the route -----------------------
# Hermes (zettlab-claw) and local-server are separate OTA packages, so a new
# tool action can meet an older server. Its router answers the unregistered
# path with a bodiless 404; every business 404 on this face carries a JSON
# {code} body. transport_error means "transient, retry" to the caller — the
# one meaning a permanently missing route must not have.

def _http_error(code, body):
    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, code, "err", None, io.BytesIO(body))
    return _boom


def test_rollback_404_without_body_is_unsupported_not_retryable(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(404, b"404 page not found")):
            out = json.loads(app_host_tool(
                {"action": "rollback", "slug": "app1", "to_version": "v1"}))
    assert out["ok"] is False and out["status"] == 404
    assert out["error"]["code"] == "unsupported"
    # The message must hand the model its fallback, not a dead end.
    assert "reload" in out["error"]["message"]


def test_publish_404_without_body_points_at_legacy_route(monkeypatch):
    """The skill's downgrade branch keys on exactly this shape. The message is
    the second, independent signpost: a model that skipped the skill's
    downgrade section must still be able to reach install/reload from the
    error alone — so it names the route, never "upgrade the device"."""
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(404, b"404 page not found")):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "install",
                "source_subdir": "runs/run-1/app1",
                "data_refresh": "static",
            }))
    assert out["ok"] is False and out["status"] == 404
    assert out["error"]["code"] == "unsupported"
    message = out["error"]["message"]
    assert ".staging" in message
    assert "install" in message and "reload" in message
    # source_subdir can be nested (runs/run-1/app1). Copying "the workspace"
    # would land the output root in .staging and bury metadata.json a few
    # levels down, where install/reload — which treat the direct child AS the
    # app root — cannot find it. The message has to name what to copy.
    assert "source_subdir" in message and "metadata.json" in message
    # Telling the model to upgrade the device is a dead end: the whole point of
    # this branch is that the device is not going to be upgraded.
    assert "升级" not in message


def test_rollback_404_with_json_body_stays_verbatim(monkeypatch):
    """A parsable 404 is the server speaking (unknown app / not the owner) —
    the unsupported mapping must never swallow it."""
    upstream = {"code": "not_found", "message": 'unknown app "app1"'}
    body = json.dumps(upstream).encode("utf-8")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(404, body)):
            out = json.loads(app_host_tool(
                {"action": "rollback", "slug": "app1", "to_version": "v1"}))
    assert out["ok"] is False and out["status"] == 404
    assert out["error"] == upstream


def test_rollback_unsupported_mapping_is_narrow(monkeypatch):
    """Only (rollback, 404, no parsable body) maps to unsupported. A wider
    match would relabel real outages as a permanent capability gap."""
    cases = [
        # Another action hitting a bodiless 404 stays transport_error.
        ({"action": "probe"}, 404),
        # rollback hitting a non-404 bodiless error stays transport_error.
        ({"action": "rollback", "slug": "app1", "to_version": "v1"}, 502),
    ]
    for args, code in cases:
        with mux_profile_scope(monkeypatch, _scope()):
            with patch("tools.apphost_tool._urlopen", _http_error(code, b"<html></html>")):
                out = json.loads(app_host_tool(args))
        assert out["ok"] is False and out["status"] == code, (args, code)
        assert out["error"]["code"] == "transport_error", (args, code)


def test_unsupported_code_alone_does_not_mean_the_device_lacks_publish(monkeypatch):
    """The skill decides "this device has no publish route" from the pair
    (code, status), never from the code alone — because `unsupported` has
    three unrelated producers. Two of them must NOT read as a missing route:
    a device with no App Host configured at all, and a metadata.json whose
    schema_version the server rejects. Both were observed live in ZET/#138."""
    with mux_profile_scope(monkeypatch, {k: "" for k in _scope()}):
        no_apphost = json.loads(app_host_tool({
            "action": "publish", "mode": "install", "source_subdir": "runs/r/a",
            "data_refresh": "static"}))
    assert no_apphost["error"]["code"] == "unsupported"
    assert no_apphost["status"] == 0, "no-App-Host must stay distinguishable by status"

    body = json.dumps({
        "code": "unsupported",
        "message": "metadata schema_version 0 is not supported",
    }).encode()
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(422, body)):
            bad_schema = json.loads(app_host_tool({
                "action": "publish", "mode": "install", "source_subdir": "runs/r/a",
            "data_refresh": "static"}))
    assert bad_schema["error"]["code"] == "unsupported"
    assert bad_schema["status"] == 422, "a fixable metadata error must stay distinguishable by status"


def test_schema_does_not_adjudicate_between_publish_and_legacy():
    """Channel choice lives in the skill, which knows what the device answered.
    When the tool's own description also ranked the channels, the model on an
    old device had two conflicting instructions and burned turns picking a
    side (ZET/#138). The description states mechanics; it does not rank."""
    text = json.dumps(APP_HOST_SCHEMA, ensure_ascii=False)
    for word in ("preferred",):
        assert word not in text.lower(), f"{word!r} ranks the channels for the skill"
    subdir = APP_HOST_SCHEMA["parameters"]["properties"]["source_subdir"]["description"]
    assert ".staging" not in subdir, (
        "a blanket .staging ban here reads as global and blocks the fallback"
    )
    # Naming `unsupported` without its status pair invites the inverse reading
    # ("unsupported ⇒ old device"), which sends a fixable 422 schema_version
    # error down a channel that rejects it again.
    description = APP_HOST_SCHEMA["description"]
    assert "404" in description and "409" in description, (
        "the description mentions unsupported; it must also bound the statuses"
    )


# --- call against a server that predates the route ---------------------------
# Same OTA-skew story as rollback/publish, with one difference: call has no
# fallback channel, and the bodiless 404 shape is identical to what a naive
# reading takes as "the app is missing" — the mapping is what keeps an old
# device from sending the model off probing other slugs and paths.

_CALL_ARGS = {"action": "call", "slug": "app1", "path": "/api/refresh",
              "http_method": "POST"}


def test_call_404_without_body_is_unsupported_not_retryable(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(404, b"404 page not found")):
            out = json.loads(app_host_tool(dict(_CALL_ARGS)))
    assert out["ok"] is False and out["status"] == 404
    assert out["error"]["code"] == "unsupported"
    # The message must break the "404 means the app is missing" reading and
    # close the door on retries — there is no fallback channel to name.
    message = out["error"]["message"]
    assert "call" in message
    assert "不代表应用不存在" in message
    # Telling the model to upgrade the device is a dead end (same rule as the
    # publish branch).
    assert "升级" not in message


def test_call_404_with_json_body_stays_verbatim(monkeypatch):
    """A parsable 404 is the server speaking — on this route it covers both
    "unknown app" and "not the owner" (deliberately the same shape, so
    existence never leaks). The unsupported mapping must never swallow it."""
    upstream = {"code": "not_found", "message": 'unknown app "app1"',
                "retryable": False}
    body = json.dumps(upstream).encode("utf-8")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(404, body)):
            out = json.loads(app_host_tool(dict(_CALL_ARGS)))
    assert out["ok"] is False and out["status"] == 404
    assert out["error"] == upstream


def test_call_unsupported_mapping_is_narrow(monkeypatch):
    # Only (call, 404, no parsable body) maps to unsupported; a bodiless
    # non-404 stays transport_error rather than a capability verdict.
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(502, b"<html></html>")):
            out = json.loads(app_host_tool(dict(_CALL_ARGS)))
    assert out["ok"] is False and out["status"] == 502
    assert out["error"]["code"] == "transport_error"


# --- call: two-layer status & retryable pass-through --------------------------

def test_call_app_level_error_is_tool_success(monkeypatch):
    """Two-layer status: data.status is the APP's answer. An app-side 500
    arrives as ok:true — the forwarding chain worked, the app answered — and
    must never be conflated with a tool failure the model would retry."""
    payload = {"status": 500, "content_type": "application/json",
               "body": {"error": "refresh source unavailable"}}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen({}, payload)):
            out = json.loads(app_host_tool(dict(_CALL_ARGS)))
    assert out["ok"] is True
    assert out["data"] == payload


@pytest.mark.parametrize("upstream,status", [
    # Transient states: the server names them retryable and pairs them with
    # Retry-After: 5.
    ({"code": "app_waking", "message": "还在启动", "retryable": True}, 503),
    ({"code": "app_updating", "message": "正在更新", "retryable": True}, 503),
    # The user pressed stop: retryable false is load-bearing — a retry loop
    # here would make the stop button decorative.
    ({"code": "app_stopped", "message": "应用已停止", "retryable": False}, 503),
    ({"code": "app_unreachable", "message": "连接失败", "retryable": False}, 502),
    ({"code": "app_response_too_large", "message": "响应过大", "retryable": False}, 502),
])
def test_call_forwarding_errors_pass_retryable_verbatim(monkeypatch, upstream, status):
    """The {code, message, retryable} error body must arrive untouched: the
    model's retry decision reads these fields, and flattening them into prose
    (or dropping retryable) severs that contract."""
    body = json.dumps(upstream).encode("utf-8")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _http_error(status, body)):
            out = json.loads(app_host_tool(dict(_CALL_ARGS)))
    assert out["ok"] is False and out["status"] == status
    assert out["error"] == upstream  # verbatim, key for key


def test_connection_error_does_not_leak_url_or_token(monkeypatch):
    scope = _scope()

    def _boom(req, timeout=None):
        raise urllib.error.URLError(
            f"connection refused: {req.full_url} token={req.get_header('X-zettlab-agent-action-token')}"
        )

    with mux_profile_scope(monkeypatch, scope):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = app_host_tool({"action": "probe"})
    parsed = json.loads(out)
    assert parsed["ok"] is False and parsed["status"] is None
    assert parsed["error"]["code"] == "transport_error"
    _assert_no_secret_leak(out, scope)


@pytest.mark.parametrize("args", [
    {"action": "install", "slug": "a1", "staging_dir": "/tmp/s",
     "data_refresh": "static"},
    {"action": "publish", "mode": "reload", "source_subdir": "runs/run-2/a1"},
])
def test_mutation_failure_is_never_auto_retried(monkeypatch, args):
    """Retry semantics belong to the calling skill; a blind tool-level retry
    would race the server's rollback-on-cancel logic."""
    scope = _scope()
    attempts = []

    def _boom(req, timeout=None):
        attempts.append(req.full_url)
        raise urllib.error.URLError("timed out")

    with mux_profile_scope(monkeypatch, scope):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = json.loads(app_host_tool(args))
    assert out["ok"] is False
    assert len(attempts) == 1, f"mutation must be attempted exactly once, got {attempts}"


def test_missing_config_returns_error(monkeypatch):
    with mux_profile_scope(monkeypatch, {k: "" for k in _scope()}):
        out = app_host_tool({"action": "probe"})
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert parsed["error"]["code"] == "unsupported"


def test_local_rejection_status_is_zero_not_null(monkeypatch):
    """status is a three-way retry contract: HTTP code = server answered;
    null = request went out but outcome unknown (idempotent resend); 0 = not
    a single byte was sent (local rejection — resending verbatim is pointless).
    Skills branch "status is null → resend"; a local rejection reported as
    null would spin that branch forever."""
    # Local rejection form 1: validation failure (never builds a request).
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen",
                   side_effect=AssertionError("must not reach the network")):
            raw_validation = app_host_tool({"action": "release_slot"})
            raw_bad_slug = app_host_tool({"action": "delete", "slug": "a/b"})
    # Local rejection form 2: credentials/base URL not configured.
    with mux_profile_scope(monkeypatch, {k: "" for k in _scope()}):
        raw_unconfigured = app_host_tool({"action": "probe"})
    for raw in (raw_validation, raw_bad_slug, raw_unconfigured):
        parsed = json.loads(raw)
        assert parsed["ok"] is False
        # Strict tier separation: 0 is falsy, so `is not None` (not
        # truthiness) is the only valid way to tell it from null — assert at
        # both the parsed and the serialized level.
        assert parsed["status"] is not None
        assert parsed["status"] == 0 and isinstance(parsed["status"], int)
        assert '"status": 0' in raw and '"status": null' not in raw


def test_local_error_requires_explicit_status():
    """A forgotten status must be a TypeError at the call site, not a silent
    null: null tells the caller "request went out, outcome unknown — resend
    idempotently", which is exactly the wrong handling for a local rejection.
    The default may not lean toward the dangerous tier."""
    with pytest.raises(TypeError):
        _local_error("invalid_request", "缺参数")  # no status → must blow up
    with pytest.raises(TypeError):
        _local_error("invalid_request", "缺参数", 0)  # positional → keyword-only
    # The three legitimate tiers all pass explicitly.
    assert json.loads(_local_error("invalid_request", "x", status=0))["status"] == 0
    assert json.loads(_local_error("transport_error", "x", status=None))["status"] is None
    assert json.loads(_local_error("transport_error", "x", status=502))["status"] == 502


# --- build_env ---------------------------------------------------------------

def test_build_env_flow_ready_when_dir_exists(monkeypatch, tmp_path):
    vendor = tmp_path / "go-vendor"
    vendor.mkdir()
    with mux_profile_scope(monkeypatch, _scope(ZETTLAB_GO_VENDOR_DIR=str(vendor)),
                           poison_environ=True):
        with patch("tools.apphost_tool._urlopen",
                   side_effect=AssertionError("build_env must not make HTTP requests")):
            out = json.loads(app_host_tool({"action": "build_env"}))
    assert out["ok"] is True
    assert out["data"] == {"vendor_dir": str(vendor), "ready": True}


def test_build_env_not_ready_when_dir_missing(monkeypatch, tmp_path):
    missing = tmp_path / "nope"
    with mux_profile_scope(monkeypatch, _scope(ZETTLAB_GO_VENDOR_DIR=str(missing))):
        out = json.loads(app_host_tool({"action": "build_env"}))
    assert out["data"]["ready"] is False
    assert out["data"]["vendor_dir"] == str(missing)


def test_build_env_not_ready_when_unset(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        out = json.loads(app_host_tool({"action": "build_env"}))
    assert out["data"] == {"vendor_dir": "", "ready": False}


# --- per-action timeouts -----------------------------------------------------

@pytest.mark.parametrize("action,args,expected_timeout", [
    # install/reload need headroom over the server pipeline (Start 30s +
    # selfCheck 5s); a client-side timeout cancels the request context and
    # triggers rollbackInstall on the server. acquire_slot pays the granted
    # slot's integrity walk before the response.
    ("install", {"slug": "a1", "staging_dir": "/tmp/s",
                 "data_refresh": "static"}, 120.0),
    ("reload", {"slug": "a1", "staging_dir": "/tmp/s"}, 120.0),
    ("publish", {"mode": "reload", "source_subdir": "runs/run-2/a1"}, 120.0),
    # No rebuild, but still stop + swap + health-check — long tier.
    ("rollback", {"slug": "a1", "to_version": "v1"}, 120.0),
    ("acquire_slot", {}, 120.0),
    ("probe", {}, 30.0),
    ("delete", {"slug": "a1"}, 30.0),
    # Deliberately the default tier: the server's wake+respond budget (~25s)
    # must expire first so failures arrive as structured error codes, not as
    # a client-side status=null transport_error.
    ("call", {"slug": "a1", "path": "/api/refresh", "http_method": "POST"}, 30.0),
])
def test_timeout_is_tiered_per_action(monkeypatch, action, args, expected_timeout):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            app_host_tool({"action": action, **args})
    assert seen["timeout"] == expected_timeout


# --- bounded waits -----------------------------------------------------------

def test_acquire_slot_uses_long_timeout_and_is_bounded(monkeypatch):
    # The server answers a queued caller immediately, but a granted slot pays
    # the ~25k-file integrity walk before responding — long tier, still
    # bounded (never unlimited).
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            app_host_tool({"action": "acquire_slot"})
    assert seen["timeout"] == 120.0


def test_default_timeout_on_other_actions(monkeypatch):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            app_host_tool({"action": "probe"})
    assert seen["timeout"] == 30.0


# --- input validation --------------------------------------------------------

@pytest.mark.parametrize("bad_slug", ["", "a/b", "../up", "a b", "a?x=1", "a#f"])
def test_bad_slug_rejected_without_http(monkeypatch, bad_slug):
    # NOTE: a raising urlopen would be swallowed by the handler's generic
    # network-error path and still yield ok=False — capture instead, and
    # assert the request was never even attempted.
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": "delete", "slug": bad_slug}))
    assert out["ok"] is False
    assert "req" not in seen, f"bad slug {bad_slug!r} reached the network"


@pytest.mark.parametrize("action,args", [
    ("release_slot", {}),
    ("install", {"slug": "app1"}),           # staging_dir missing
    ("install", {"staging_dir": "/tmp/s"}),  # slug missing
    ("reload", {"slug": "app1"}),            # staging_dir missing
    ("publish", {"source_subdir": "runs/run-1/app1"}),  # mode missing
    ("publish", {"mode": "install"}),         # source_subdir missing
    # to_version missing: an untargeted rollback is a symmetric swap, so a
    # retry after a lost response would undo the undo — the tool refuses to
    # send one even though the server would accept it.
    ("rollback", {"slug": "app1"}),
    ("rollback", {"slug": "app1", "to_version": "   "}),  # whitespace is not a target
    ("lifecycle", {"slug": "app1"}),         # lifecycle_action missing
    ("lifecycle", {"slug": "app1", "lifecycle_action": "explode"}),
    ("call", {"path": "/api/x", "http_method": "GET"}),   # slug missing
    ("call", {"slug": "app1", "http_method": "GET"}),     # path missing
    ("call", {"slug": "app1", "path": "/api/x"}),         # http_method missing
    ("call", {"slug": "app1", "path": "/api/x", "http_method": "FETCH"}),
    # HEAD/OPTIONS are real methods but not app domain verbs — not offered.
    ("call", {"slug": "app1", "path": "/api/x", "http_method": "HEAD"}),
])
def test_missing_required_params_rejected_without_http(monkeypatch, action, args):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is False
    assert "req" not in seen, f"{action} with {args} reached the network"


# --- server route table is the fixture ---------------------------------------

# Verbatim route table of the internal face (RegisterInternalRoutes,
# zettlab-local-server internal/apphost/handler/internal.go). Path params are
# templated. This is the source of truth the action map must stay inside —
# the recover action shipped against a route that only exists on the JWT
# member face, and the tests were blind to it because fixture and
# implementation encoded the same wrong assumption.
_SERVER_INTERNAL_ROUTES = {
    ("GET", "/storage"),
    ("POST", "/buildslot"),
    ("DELETE", "/buildslot/{token}"),
    ("POST", "/publish"),
    ("POST", "/install"),
    ("GET", ""),
    ("POST", "/{name}/reload"),
    ("POST", "/{name}/rollback"),
    ("DELETE", "/{name}"),
    ("POST", "/{name}/lifecycle"),
    ("GET", "/{name}/logs"),
    ("POST", "/{name}/call"),
}


def _route_template(method, path):
    """Normalize a concrete request path back to its route template."""
    path = path.split("?", 1)[0]
    parts = path.split("/")
    if len(parts) >= 2 and parts[1] == "buildslot" and len(parts) == 3:
        parts[2] = "{token}"
    elif len(parts) >= 2 and parts[1] not in (
        "storage", "buildslot", "publish", "install", ""
    ):
        parts[1] = "{name}"
    return method, "/".join(parts)


def test_every_action_routes_inside_server_route_table(monkeypatch):
    """Generalized guard: each HTTP action's (method, path) must land on a
    route the server actually registers. Catches the next wrong-path action
    at authoring time instead of as a live 404."""
    for action, args in _ALL_HTTP_ACTION_ARGS:
        seen = {}
        with mux_profile_scope(monkeypatch, _scope()):
            completion_status = 204 if action in {"release_slot", "delete"} else 200
            with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, status=completion_status)):
                out = json.loads(app_host_tool({"action": action, **args}))
        assert out["ok"] is True, action
        req = seen["req"]
        rel = req.full_url[len(_BASE_URL):]
        assert _route_template(req.get_method(), rel) in _SERVER_INTERNAL_ROUTES, (
            f"{action} -> {req.get_method()} {rel} is not a server route"
        )


def test_recover_is_not_an_action():
    # The internal face deliberately has no recover route (an action token
    # authenticates one agent, not the device); recovery lives on the client.
    advertised = APP_HOST_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "recover" not in advertised
    out = json.loads(app_host_tool({"action": "recover", "slug": "app1"}))
    assert out["ok"] is False
    assert out["error"]["code"] == "invalid_request"


# --- credential never leaves loopback ----------------------------------------

@pytest.mark.parametrize("bad_base", [
    "https://127.0.0.1:18080/api/v1/internal/apps",  # internal face is plain http
    "http://192.168.1.10:18080/api/v1/internal/apps",  # not loopback
    "http://evil.example/api/v1/internal/apps",
    "http://127.attacker.example/api/v1/internal/apps",  # prefix trick
])
def test_non_loopback_base_url_closes_gate_and_refuses_calls(monkeypatch, bad_base):
    """The action token must never ride to a non-loopback endpoint: a
    repointed base URL closes the gate and every call fails locally (status
    0) without any request being attempted."""
    seen = {}
    with mux_profile_scope(monkeypatch, _scope(ZET_APPHOST_BASE_URL=bad_base)):
        assert _check_app_host() is False
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is False and out["status"] == 0
    assert "req" not in seen


@pytest.mark.parametrize("good_base", [
    "http://127.0.0.1:18080/api/v1/internal/apps",
    "http://localhost:18080/api/v1/internal/apps",
    "http://[::1]:18080/api/v1/internal/apps",
])
def test_loopback_base_urls_pass_the_gate(monkeypatch, good_base):
    with mux_profile_scope(monkeypatch, _scope(ZET_APPHOST_BASE_URL=good_base)):
        assert _check_app_host() is True


def test_requests_bypass_environment_proxies(monkeypatch):
    """The transport must refuse env proxies (HTTP_PROXY/ALL_PROXY would
    forward the credentialed loopback request off-box): the tool goes through
    the no-proxy opener, never the global urlopen."""
    import urllib.request as _ur
    from tools.apphost_tool import _NO_PROXY_OPENER

    # build_opener(ProxyHandler({})) displaces the DEFAULT ProxyHandler (which
    # reads HTTP_PROXY/ALL_PROXY from the environment); the empty one defines
    # no <scheme>_open methods so it never joins the handler chain. Net
    # effect, and the property asserted here: no proxy handler at all.
    assert not [h for h in _NO_PROXY_OPENER.handlers
                if isinstance(h, _ur.ProxyHandler)]

    # Behavior-level: plant a distinguishable response on each transport and
    # assert the result came through the no-proxy opener. Deliberately does
    # NOT patch _urlopen itself (that would replace the very code under test)
    # and does NOT use a raising sentinel (the handler's generic except would
    # translate it into an ordinary failure).
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("urllib.request.urlopen",
                   return_value=_Resp({"via": "global-urlopen"})), \
             patch.object(_NO_PROXY_OPENER, "open",
                          return_value=_Resp({"via": "no-proxy-opener"})):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is True
    assert out["data"] == {"via": "no-proxy-opener"}


def test_redirects_are_not_followed_and_token_stays_home(monkeypatch):
    """Real-socket reproduction of the redirect leak: base (loopback) answers
    302 pointing at another origin. The loopback check only constrains the
    first hop, so the transport itself must refuse to follow — the redirect
    target must receive NOTHING (no request, no token), and the tool must
    report the 3xx as a transport_error. Deliberately unmocked: the redirect
    decision lives inside the opener, which a patched _urlopen would bypass."""
    import http.server
    import threading

    target_hits = []

    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            target_hits.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            pass

    target_srv = http.server.HTTPServer(("127.0.0.1", 0), Target)
    target_url = f"http://127.0.0.1:{target_srv.server_port}"

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target_url + "/steal")
            self.end_headers()

        def log_message(self, *_):
            pass

    redirect_srv = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
    base = f"http://127.0.0.1:{redirect_srv.server_port}/api/v1/internal/apps"
    for srv in (target_srv, redirect_srv):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with mux_profile_scope(monkeypatch, _scope(ZET_APPHOST_BASE_URL=base)):
            out = json.loads(app_host_tool({"action": "probe"}))
    finally:
        for srv in (target_srv, redirect_srv):
            srv.shutdown()
            srv.server_close()

    assert out["ok"] is False
    assert out["status"] == 302
    assert out["error"]["code"] == "transport_error"
    assert target_hits == [], f"redirect target was contacted: {target_hits}"


# --- staging_dir precheck ----------------------------------------------------

@pytest.mark.parametrize("bad_staging", [
    "relative/path",
    "/tmp/stage/../../../etc",
    "/tmp/stage\nX-Injected: 1",
    "/tmp/stage\x00",
    "/" + "a" * 2000,
])
def test_malformed_staging_dir_rejected_without_http(monkeypatch, bad_staging):
    """String-level precheck: obviously-malformed staging paths never ride a
    credentialed request (the server's validateStagingPath stays the
    authoritative gate for containment and symlinks)."""
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool(
                {"action": "install", "slug": "app1", "staging_dir": bad_staging,
                 "data_refresh": "static"}
            ))
    assert out["ok"] is False and out["status"] == 0
    assert out["error"]["code"] == "invalid_request"
    assert "req" not in seen


@pytest.mark.parametrize("bad_source", [
    "",
    ".",
    "../another-agent/app",
    "runs/../another-agent/app",
    "/volume1/subvol/agents/data/agent-a/output/app",
    "runs\\run-1\\app",
    "runs/run-1/app\nX-Injected: 1",
    "runs/run-1/app\x00",
    "a" * 2000,
])
def test_malformed_publish_source_rejected_without_http(monkeypatch, bad_source):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "install",
                "source_subdir": bad_source,
            }))
    assert out["ok"] is False and out["status"] == 0
    assert out["error"]["code"] == "invalid_request"
    assert "req" not in seen


@pytest.mark.parametrize("bad_path", [
    "api/refresh",                        # not rooted at the app
    "//evil.example/steal",               # host-relative URL form
    "http://127.0.0.1:9/x",               # full URL
    "/redirect?to=https://x",             # embedded absolute URL anywhere
    "/api/../internal",                   # traversal segment
    "/api/refresh\nX-Injected: 1",
    "/api/refresh\x00",
    "/api/\x1bcontrol",
    "/" + "a" * 2000,
])
def test_malformed_call_path_rejected_without_http(monkeypatch, bad_path):
    """String-level precheck: a path that is really a URL, a traversal, or
    carries control characters never rides a credentialed request (the server
    stays the authoritative gate)."""
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({**_CALL_ARGS, "path": bad_path}))
    assert out["ok"] is False and out["status"] == 0
    assert out["error"]["code"] == "invalid_request"
    assert "req" not in seen


def test_call_http_method_is_case_normalized(monkeypatch):
    # "post" is unambiguous — normalize instead of burning a model turn on a
    # case correction. The wire form is canonical uppercase.
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({**_CALL_ARGS, "http_method": "post"}))
    assert out["ok"] is True
    assert json.loads(seen["req"].data.decode("utf-8"))["method"] == "POST"


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/refresh"),
        ("POST", "/api/delete-account"),
        ("POST", "/api/refresh-all"),
        ("PUT", "/api/refresh"),
        ("PATCH", "/api/config"),
        ("DELETE", "/api/items/1"),
    ],
)
def test_call_rejects_writes_outside_the_refresh_capability(
    monkeypatch, method, path
):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({
                "action": "call",
                "slug": "app1",
                "path": path,
                "http_method": method,
            }))
    assert out["ok"] is False
    assert out["status"] == 0
    assert "req" not in seen


# --- response-size caps ------------------------------------------------------

def test_error_body_read_is_capped(monkeypatch):
    """The error path must read with the same cap as the success path — an
    oversized error body degrades to transport_error instead of ballooning
    memory on a 2 GB shared device. Asserting the outcome alone would go
    green even without the cap (an over-long body degrades either way), so
    the read AMOUNTS are recorded and bounded."""
    from tools.apphost_tool import _MAX_RESPONSE_BYTES

    read_amounts = []

    class _RecordingBody(io.BytesIO):
        def read(self, amt=None):
            read_amounts.append(amt)
            return super().read(amt)

    huge = b"x" * (_MAX_RESPONSE_BYTES + 4096)

    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 507, "Insufficient Storage",
                                     None, _RecordingBody(huge))

    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is False and out["status"] == 507
    assert out["error"]["code"] == "transport_error"
    assert read_amounts, "error body was never read"
    assert all(amt is not None and amt <= _MAX_RESPONSE_BYTES + 1
               for amt in read_amounts), read_amounts


def test_response_cap_exceeds_server_logs_cap(monkeypatch):
    """The server caps a single logs response at 512 KiB; a smaller client
    cap makes every long-log fetch fail as status=200 + transport_error,
    which the skill reads as a transient outage and retries forever."""
    from tools.apphost_tool import _MAX_RESPONSE_BYTES

    assert _MAX_RESPONSE_BYTES > 512 * 1024
    # Behavior-level: a server-cap-sized text body must come back as success
    # (the delivery layer then tail-truncates it — that is labeled, not an
    # error; see test_oversized_text_payload_is_tail_truncated_and_labeled).
    big_log = ("L" * 1024 + "\n") * 512  # ~512 KiB
    resp = _RawResp(200, big_log.encode("utf-8"), content_type="text/plain")
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=resp):
            out = json.loads(app_host_tool({"action": "logs", "slug": "app1"}))
    assert out["ok"] is True
    assert out["data"]["truncated"] is True
    assert out["data"]["total_chars"] == len(big_log)


# --- delegated children never inherit device-state verbs ---------------------

def test_delegated_children_never_get_app_host(monkeypatch):
    """Delegated workers propagate the parent's secret scope, so without a
    block an anonymous child could install/delete/restart device applications
    with the parent's action token. Real path, with a control: the SAME
    enabled set and credentials expose app_host to the parent, and the
    child-assembly deny toolsets strip exactly it — proving the absence below
    is the blocklist's doing, not a closed gate."""
    import model_tools
    from hermes_cli.tools_config import _get_platform_tools
    from tools.delegate_tool import DELEGATE_BLOCKED_TOOLS, _blocked_toolsets_for_role

    assert "app_host" in DELEGATE_BLOCKED_TOOLS  # structural aid, not the proof

    device_config = {"platform_toolsets": {"zet_agent": ["hermes-zet-agent", "cronjob"]}}
    enabled = sorted(_get_platform_tools(
        device_config, "zet_agent", include_default_mcp_servers=False
    ))
    with mux_profile_scope(monkeypatch, _scope()):
        parent_names = {
            d["function"]["name"]
            for d in model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
        child_names = {
            d["function"]["name"]
            for d in model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                disabled_toolsets=_blocked_toolsets_for_role("worker"),
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
    assert "app_host" in parent_names  # control: reachable before the block
    assert "app_host" not in child_names


# --- results are attacker-influenced data ------------------------------------

def test_app_host_results_are_marked_untrusted():
    # App logs (and app-shaped error messages) can carry third-party content;
    # the dispatch layer must wrap app_host output in the untrusted-result
    # delimiters like web_search/browser_* results.
    from agent.tool_dispatch_helpers import _is_untrusted_tool

    assert _is_untrusted_tool("app_host") is True


def test_schema_actions_match_handler():
    # Every schema-advertised action must be handled (no dead enum entries):
    # unknown actions fail, advertised ones never return the unknown-action error.
    advertised = APP_HOST_SCHEMA["parameters"]["properties"]["action"]["enum"]
    for action in advertised:
        out = json.loads(app_host_tool({"action": action}))
        error = out.get("error") or {}
        assert "未知动作" not in str(error.get("message", "")), action


# --- reachability ------------------------------------------------------------
# The model can only call what the schema declares. Handling an action in
# _build_request is not enough: an action missing from the enum is invisible,
# and the model works around it — which is exactly how undo ended up being
# "recompile the app with the old content" instead of one step back.

def test_schema_declares_every_action_it_handles():
    declared = set(APP_HOST_SCHEMA["parameters"]["properties"]["action"]["enum"])
    for action in (
        "publish", "rollback", "reload", "install", "list", "delete",
        "lifecycle", "logs", "call",
    ):
        assert action in declared, f"{action} is handled but not offered to the model"


def test_schema_declares_the_arguments_undo_depends_on():
    props = APP_HOST_SCHEMA["parameters"]["properties"]
    # Without to_version an undo cannot be retried safely: the swap is
    # symmetric, so repeating an untargeted one swaps the app forward again.
    assert "to_version" in props
    # Without note the archived version has no description, and the user is
    # asked to undo something the host can only identify by a timestamp.
    assert "note" in props


def test_schema_requires_data_refresh_on_both_install_paths():
    description = APP_HOST_SCHEMA["parameters"]["properties"]["data_refresh"]["description"]
    assert "publish(mode=install)" in description
    assert "legacy action=install" in description


def test_undo_is_described_where_the_model_reads_it():
    text = APP_HOST_SCHEMA["description"]
    assert "rollback" in text
    assert "prev_version_id" in text, "the model has to be told where to get to_version"


def test_call_is_described_where_the_model_reads_it():
    """The description is the only place the model learns call's retry
    discipline — the error codes come from the server, but which ones to obey
    without retrying has to be said up front."""
    text = APP_HOST_SCHEMA["description"]
    # Transient vs terminal must both be named…
    assert "app_updating" in text and "app_waking" in text
    # …and app_stopped must be tied to a no-retry instruction (the user
    # pressed stop; a retry loop would make that button decorative).
    assert "app_stopped" in text
    assert "never retry" in text
    # The two-layer status contract: an app-side error is not a tool failure.
    assert "data.status" in text
    props = APP_HOST_SCHEMA["parameters"]["properties"]
    for param in ("path", "http_method", "body"):
        assert param in props, f"call's {param} is handled but not declared"
    assert list(_CALL_HTTP_METHODS) == props["http_method"]["enum"]


# --- creation provenance (session key) ---------------------------------------

_SESSION_KEY = "zettlab:usr-1f2e3d:main:0"


def _routed_body(monkeypatch, action_args):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope(), poison_environ=True):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool(action_args))
    assert out["ok"] is True
    req = seen["req"]
    return json.loads(req.data.decode("utf-8")) if req.data else None


def test_publish_install_carries_stable_session_key(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", _SESSION_KEY)
    monkeypatch.setenv("HERMES_SESSION_ID", "api-rotated-tip")
    body = _routed_body(monkeypatch, {
        "action": "publish", "mode": "install", "source_subdir": "runs/run-1/app1",
        "data_refresh": "static",
    })
    assert body["session_id"] == _SESSION_KEY


def test_publish_install_uses_current_session_build_when_path_is_omitted(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", _SESSION_KEY)
    body = _routed_body(monkeypatch, {
        "action": "publish", "mode": "install", "data_refresh": "static",
    })
    assert body["mode"] == "install"
    assert "source_subdir" not in body
    assert body["session_id"] == _SESSION_KEY


def test_publish_reload_never_rewrites_creation_provenance(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", _SESSION_KEY)
    body = _routed_body(monkeypatch, {
        "action": "publish", "mode": "reload", "source_subdir": "runs/run-2/app1",
    })
    assert "session_id" not in body


@pytest.mark.parametrize(
    "data_refresh", ["static", "external_unconfirmed", "user_declined"]
)
def test_legacy_install_carries_session_key_and_data_refresh(monkeypatch, data_refresh):
    monkeypatch.setenv("HERMES_SESSION_KEY", _SESSION_KEY)
    body = _routed_body(monkeypatch, {
        "action": "install", "staging_dir": "/tmp/stage", "slug": "app1",
        "data_refresh": data_refresh,
    })
    assert body["session_id"] == _SESSION_KEY
    assert body["data_refresh"] == data_refresh


def test_rotating_session_id_is_not_provenance(monkeypatch):
    # HERMES_SESSION_ID rotates on context compaction: recording it would
    # name a session that stops being findable mid-conversation. Without the
    # stable key there must be no session_id at all.
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", "api-rotated-tip")
    body = _routed_body(monkeypatch, {
        "action": "publish", "mode": "install", "source_subdir": "runs/run-1/app1",
        "data_refresh": "static",
    })
    assert "session_id" not in body


# --- data_refresh: installing forces an answer -------------------------------
# Five device runs shipped a dashboard with a manual button after the user
# asked for a daily fetch. Every one of them had read the skill text that says
# to ask. Skill text loses arguments with other skill text; a required argument
# does not, so the decision moved into the tool call itself.

def _never_called(req, timeout=None):
    raise AssertionError("the tool must reject this before any HTTP call")


def test_install_without_data_refresh_never_reaches_the_network(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _never_called):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "install",
                "source_subdir": "runs/run-1/app1",
            }))
    assert out["ok"] is False
    assert "data_refresh" in out["error"]["message"]


def test_legacy_install_without_data_refresh_never_reaches_the_network(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _never_called):
            out = json.loads(app_host_tool({
                "action": "install",
                "staging_dir": "/tmp/stage",
                "slug": "app1",
            }))
    assert out["ok"] is False
    assert "data_refresh" in out["error"]["message"]


@pytest.mark.parametrize("value", ["", "auto", "yes", "AUTO_CONFIGURED", "true"])
def test_install_rejects_values_outside_the_enum(monkeypatch, value):
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _never_called):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "install",
                "source_subdir": "runs/run-1/app1",
                "data_refresh": value,
            }))
    assert out["ok"] is False
    assert "data_refresh" in out["error"]["message"]


@pytest.mark.parametrize(
    "value", ["static", "external_unconfirmed", "user_declined"])
def test_install_forwards_every_accepted_answer(monkeypatch, value):
    seen = {}

    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "install",
                "source_subdir": "runs/run-1/app1",
                "data_refresh": value,
            }))
    assert out["ok"] is True
    # The server records "the user asked for this", so it has to arrive intact.
    assert json.loads(seen["req"].data.decode("utf-8"))["data_refresh"] == value


def test_reload_does_not_ask_again(monkeypatch):
    """Reload changes code on an app that already answered this at install."""
    seen = {}

    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({
                "action": "publish",
                "mode": "reload",
                "source_subdir": "runs/run-2/app1",
                "note": "Footer 加了一个链接",
            }))
    assert out["ok"] is True
    assert "data_refresh" not in json.loads(seen["req"].data.decode("utf-8"))


# --- 就地创建 / 就地发布 -------------------------------------------------------

def test_prepare_asks_the_platform_for_the_app_directory(monkeypatch):
    """一个应用从创建到退役只有一个目录，而那个目录的位置是平台的口径。

    模型不再自己在 output 下拼一个带哈希的工地路径——它问平台要，平台建好目录
    和版本库再把绝对路径交出来。这条钉住线上的形状：POST /prepare，只带 slug。
    """
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        # /prepare 回的是裸对象，不是 {code,data} 信封（服务端 c.JSON(200, prepared)）。
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, {
            "dir": "/volume1/subvol/apps/mood-journal",
            "installed": False,
            "build_output": "bin/app.new",
        })):
            out = json.loads(app_host_tool({"action": "prepare", "slug": "mood-journal"}))
    assert out["ok"] is True
    assert seen["req"].get_method() == "POST"
    assert seen["req"].full_url.endswith("/prepare")
    assert json.loads(seen["req"].data.decode()) == {"slug": "mood-journal"}
    assert out["data"]["dir"] == "/volume1/subvol/apps/mood-journal"
    # 编译输出由平台指定：绝不能是 bin/app——改一个在跑的应用时，那底下是正在
    # 被执行的文件。
    assert out["data"]["build_output"] == "bin/app.new"


def test_prepare_refuses_without_a_slug(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        out = json.loads(app_host_tool({"action": "prepare"}))
    assert out["ok"] is False
    assert "slug" in out["error"]["message"]


def test_in_place_publish_sends_no_source_subdir(monkeypatch):
    """就地发布不拷贝任何东西，所以线上没有 source_subdir 可言，改成报应用名。"""
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, {"code": 200, "data": {}})):
            out = json.loads(app_host_tool({
                "action": "publish", "mode": "reload",
                "in_place": True, "slug": "mood-journal", "note": "改了首页",
            }))
    assert out["ok"] is True
    body = json.loads(seen["req"].data.decode())
    assert body["in_place"] is True and body["slug"] == "mood-journal"
    assert "source_subdir" not in body
    assert body["note"] == "改了首页"


def test_in_place_publish_requires_a_slug(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        out = json.loads(app_host_tool({"action": "publish", "mode": "reload", "in_place": True}))
    assert out["ok"] is False
    assert "slug" in out["error"]["message"]


def test_discard_sends_only_the_slug(monkeypatch):
    """「改砸了退回去」只需要说清是哪个应用——退到哪一版由平台的指针决定，
    不给模型任何「退到我说的那一版」的余地。"""
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen, {"discarded": True})):
            out = json.loads(app_host_tool({"action": "discard", "slug": "mood-journal"}))
    assert out["ok"] is True
    assert seen["req"].get_method() == "POST"
    assert seen["req"].full_url.endswith("/discard")
    assert json.loads(seen["req"].data.decode()) == {"slug": "mood-journal"}


def test_discard_refuses_without_a_slug(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()):
        out = json.loads(app_host_tool({"action": "discard"}))
    assert out["ok"] is False
    assert "slug" in out["error"]["message"]
