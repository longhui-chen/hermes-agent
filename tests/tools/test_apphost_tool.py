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


@pytest.mark.parametrize("action,args,method,path,body", [
    ("probe", {}, "GET", "/storage", None),
    ("list", {}, "GET", "?mine=1", None),
    ("acquire_slot", {}, "POST", "/buildslot", None),
    ("release_slot", {"slot_token": "s1"}, "DELETE", "/buildslot/s1", None),
    ("install", {"staging_dir": "/tmp/stage", "slug": "app1"}, "POST", "/install",
     {"staging_dir": "/tmp/stage", "slug": "app1"}),
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
])
def test_action_routing_flow(monkeypatch, action, args, method, path, body):
    seen = {}
    with mux_profile_scope(monkeypatch, _scope(), poison_environ=True):
        with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
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


# --- 2xx is success regardless of body ---------------------------------------

# Every HTTP action with the minimal args to reach the network layer.
_ALL_HTTP_ACTION_ARGS = [
    ("probe", {}),
    ("list", {}),
    ("acquire_slot", {}),
    ("release_slot", {"slot_token": "s1"}),
    ("install", {"staging_dir": "/tmp/s", "slug": "app1"}),
    ("reload", {"slug": "app1", "staging_dir": "/tmp/s"}),
    ("rollback", {"slug": "app1", "to_version": "v1"}),
    ("delete", {"slug": "app1"}),
    ("lifecycle", {"slug": "app1", "lifecycle_action": "restart"}),
    ("logs", {"slug": "app1"}),
]


@pytest.mark.parametrize("action,args", _ALL_HTTP_ACTION_ARGS)
def test_2xx_empty_body_is_success_for_every_action(monkeypatch, action, args):
    """The upstream deliberately answers 204 with no body (release_slot
    always; delete idempotently). A 2xx must never fall into the error
    branch — flagging it as transport_error reported every successful
    release/delete as a failure on a real device."""
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=_RawResp(204)):
            out = json.loads(app_host_tool({"action": action, **args}))
    assert out["ok"] is True
    assert out["data"] == {}


@pytest.mark.parametrize("status", [200, 201, 202, 204])
def test_2xx_success_tier_is_the_range_not_specific_codes(monkeypatch, status):
    # The tier test must be "status is 2xx", not an enumeration of codes:
    # a future 200-empty-body or 202 must not degrade into an error.
    with mux_profile_scope(monkeypatch, _scope()):
        with patch("tools.apphost_tool._urlopen", return_value=_RawResp(status)):
            out = json.loads(app_host_tool({"action": "release_slot", "slot_token": "s1"}))
    assert out["ok"] is True


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
            out = app_host_tool({"action": "install", "slug": "a1", "staging_dir": "/tmp/s"})
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


def test_install_failure_is_never_auto_retried(monkeypatch):
    """Retry semantics belong to the calling skill; a blind tool-level retry
    would race the server's rollback-on-cancel logic."""
    scope = _scope()
    attempts = []

    def _boom(req, timeout=None):
        attempts.append(req.full_url)
        raise urllib.error.URLError("timed out")

    with mux_profile_scope(monkeypatch, scope):
        with patch("tools.apphost_tool._urlopen", _boom):
            out = json.loads(app_host_tool({"action": "install", "slug": "a1", "staging_dir": "/tmp/s"}))
    assert out["ok"] is False
    assert len(attempts) == 1, f"install must be attempted exactly once, got {attempts}"


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
    ("install", {"slug": "a1", "staging_dir": "/tmp/s"}, 120.0),
    ("reload", {"slug": "a1", "staging_dir": "/tmp/s"}, 120.0),
    # No rebuild, but still stop + swap + health-check — long tier.
    ("rollback", {"slug": "a1", "to_version": "v1"}, 120.0),
    ("acquire_slot", {}, 120.0),
    ("probe", {}, 30.0),
    ("delete", {"slug": "a1"}, 30.0),
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
    # to_version missing: an untargeted rollback is a symmetric swap, so a
    # retry after a lost response would undo the undo — the tool refuses to
    # send one even though the server would accept it.
    ("rollback", {"slug": "app1"}),
    ("rollback", {"slug": "app1", "to_version": "   "}),  # whitespace is not a target
    ("lifecycle", {"slug": "app1"}),         # lifecycle_action missing
    ("lifecycle", {"slug": "app1", "lifecycle_action": "explode"}),
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
    ("POST", "/install"),
    ("GET", ""),
    ("POST", "/{name}/reload"),
    ("POST", "/{name}/rollback"),
    ("DELETE", "/{name}"),
    ("POST", "/{name}/lifecycle"),
    ("GET", "/{name}/logs"),
}


def _route_template(method, path):
    """Normalize a concrete request path back to its route template."""
    path = path.split("?", 1)[0]
    parts = path.split("/")
    if len(parts) >= 2 and parts[1] == "buildslot" and len(parts) == 3:
        parts[2] = "{token}"
    elif len(parts) >= 2 and parts[1] not in ("storage", "buildslot", "install", ""):
        parts[1] = "{name}"
    return method, "/".join(parts)


def test_every_action_routes_inside_server_route_table(monkeypatch):
    """Generalized guard: each HTTP action's (method, path) must land on a
    route the server actually registers. Catches the next wrong-path action
    at authoring time instead of as a live 404."""
    for action, args in _ALL_HTTP_ACTION_ARGS:
        seen = {}
        with mux_profile_scope(monkeypatch, _scope()):
            with patch("tools.apphost_tool._urlopen", _capture_urlopen(seen)):
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
                {"action": "install", "slug": "app1", "staging_dir": bad_staging}
            ))
    assert out["ok"] is False and out["status"] == 0
    assert out["error"]["code"] == "invalid_request"
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
                enabled_toolsets=enabled, quiet_mode=True
            )
        }
        child_names = {
            d["function"]["name"]
            for d in model_tools.get_tool_definitions(
                enabled_toolsets=enabled,
                disabled_toolsets=_blocked_toolsets_for_role("worker"),
                quiet_mode=True,
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
    for action in ("rollback", "reload", "install", "list", "delete", "lifecycle", "logs"):
        assert action in declared, f"{action} is handled but not offered to the model"


def test_schema_declares_the_arguments_undo_depends_on():
    props = APP_HOST_SCHEMA["parameters"]["properties"]
    # Without to_version an undo cannot be retried safely: the swap is
    # symmetric, so repeating an untargeted one swaps the app forward again.
    assert "to_version" in props
    # Without note the archived version has no description, and the user is
    # asked to undo something the host can only identify by a timestamp.
    assert "note" in props


def test_undo_is_described_where_the_model_reads_it():
    text = APP_HOST_SCHEMA["description"]
    assert "rollback" in text
    assert "prev_version_id" in text, "the model has to be told where to get to_version"
