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
    _check_app_host,
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


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps(self._payload).encode("utf-8")


def _capture_urlopen(seen, payload=None):
    def _open(req, timeout=None):
        seen["req"] = req
        seen["timeout"] = timeout
        return _Resp(payload if payload is not None else {"code": 200, "data": {}})

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
        with patch("urllib.request.urlopen", _capture_urlopen(seen, {"apps": []})):
            out = app_host_tool({"action": "list"})
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["ok"] is True and parsed["data"] == {"apps": []}
    req = seen["req"]
    assert req.full_url == _BASE_URL
    assert req.get_header("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)


def test_profile_scope_flow_works_with_empty_environ(monkeypatch):
    scope = _scope()
    seen = {}
    with mux_profile_scope(monkeypatch, scope):  # scope keys purged from env
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": "probe"}))
    assert out["ok"] is True
    assert seen["req"].full_url == _BASE_URL + "/storage"


@pytest.mark.parametrize("action,args,method,path,body", [
    ("probe", {}, "GET", "/storage", None),
    ("list", {}, "GET", "", None),
    ("acquire_slot", {}, "POST", "/buildslot", None),
    ("release_slot", {"slot_token": "s1"}, "DELETE", "/buildslot/s1", None),
    ("install", {"staging_dir": "/tmp/stage", "slug": "app1"}, "POST", "/install",
     {"staging_dir": "/tmp/stage", "slug": "app1"}),
    ("reload", {"slug": "app1", "staging_dir": "/tmp/stage"}, "POST", "/app1/reload",
     {"staging_dir": "/tmp/stage"}),
    ("delete", {"slug": "app1"}, "DELETE", "/app1", None),
    ("recover", {"slug": "app1"}, "POST", "/app1/recover", None),
    ("lifecycle", {"slug": "app1", "lifecycle_action": "restart"}, "POST",
     "/app1/lifecycle", {"action": "restart"}),
    ("logs", {"slug": "app1", "tail": 50}, "GET", "/app1/logs?tail=50", None),
])
def test_action_routing_flow(monkeypatch, action, args, method, path, body):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope(), poison_environ=True):
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
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
        with patch("urllib.request.urlopen", _capture_urlopen({})):
            success = app_host_tool({"action": "list"})
    unknown = app_host_tool({"action": "definitely_not_an_action"})
    missing = app_host_tool({"action": "release_slot"})
    for out in (success, unknown, missing):
        assert isinstance(out, str) and not isinstance(out, dict)
        json.loads(out)  # must be valid JSON


# --- error paths never leak secrets -----------------------------------------

def _assert_no_secret_leak(out, scope):
    for value in scope.values():
        assert value not in out, f"secret value leaked into tool output: {out}"
    assert "127.0.0.1:18080" not in out  # netloc of the base URL


def test_http_error_reports_status_without_leaking(monkeypatch):
    scope = _scope()
    upstream = json.dumps({
        "code": 507,
        "data": {"detail": f"insufficient storage at {_BASE_URL}/install"},
    }).encode("utf-8")

    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 507, "Insufficient Storage", None, io.BytesIO(upstream)
        )

    with mux_profile_scope(monkeypatch, scope):
        with patch("urllib.request.urlopen", _boom):
            out = app_host_tool({"action": "install", "slug": "a1", "staging_dir": "/tmp/s"})
    parsed = json.loads(out)
    assert parsed["ok"] is False and parsed["status"] == 507
    _assert_no_secret_leak(out, scope)


def test_connection_error_does_not_leak_url_or_token(monkeypatch):
    scope = _scope()

    def _boom(req, timeout=None):
        raise urllib.error.URLError(
            f"connection refused: {req.full_url} token={req.get_header('X-zettlab-agent-action-token')}"
        )

    with mux_profile_scope(monkeypatch, scope):
        with patch("urllib.request.urlopen", _boom):
            out = app_host_tool({"action": "probe"})
    parsed = json.loads(out)
    assert parsed["ok"] is False and parsed["status"] is None
    _assert_no_secret_leak(out, scope)


def test_missing_config_returns_error(monkeypatch):
    with mux_profile_scope(monkeypatch, {k: "" for k in _scope()}):
        out = app_host_tool({"action": "probe"})
    parsed = json.loads(out)
    assert parsed["ok"] is False


# --- build_env ---------------------------------------------------------------

def test_build_env_flow_ready_when_dir_exists(monkeypatch, tmp_path):
    vendor = tmp_path / "go-vendor"
    vendor.mkdir()
    with mux_profile_scope(monkeypatch, _scope(ZETTLAB_GO_VENDOR_DIR=str(vendor)),
                           poison_environ=True):
        with patch("urllib.request.urlopen",
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


# --- bounded waits -----------------------------------------------------------

def test_acquire_slot_wait_is_bounded(monkeypatch):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
            app_host_tool({"action": "acquire_slot", "wait_seconds": 99999})
    assert seen["timeout"] == 300  # capped, never unbounded
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
            app_host_tool({"action": "acquire_slot"})
    assert seen["timeout"] == 30.0  # default


def test_default_timeout_on_other_actions(monkeypatch):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
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
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": "delete", "slug": bad_slug}))
    assert out["ok"] is False
    assert "req" not in seen, f"bad slug {bad_slug!r} reached the network"


@pytest.mark.parametrize("action,args", [
    ("release_slot", {}),
    ("install", {"slug": "app1"}),           # staging_dir missing
    ("install", {"staging_dir": "/tmp/s"}),  # slug missing
    ("reload", {"slug": "app1"}),            # staging_dir missing
    ("lifecycle", {"slug": "app1"}),         # lifecycle_action missing
    ("lifecycle", {"slug": "app1", "lifecycle_action": "explode"}),
])
def test_missing_required_params_rejected_without_http(monkeypatch, action, args):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("urllib.request.urlopen", _capture_urlopen(seen)):
            out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is False
    assert "req" not in seen, f"{action} with {args} reached the network"


def test_schema_actions_match_handler():
    # Every schema-advertised action must be handled (no dead enum entries):
    # unknown actions fail, advertised ones never return the unknown-action error.
    advertised = APP_HOST_SCHEMA["parameters"]["properties"]["action"]["enum"]
    for action in advertised:
        out = json.loads(app_host_tool({"action": action}))
        assert "未知动作" not in (out.get("error") or ""), action
