import json
import sys
from types import SimpleNamespace

import pytest
import requests

from agent import secret_scope
from gateway.session_context import clear_session_vars, set_session_vars
from tools import browser_backend_router as router, browser_tool


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, post):
        self.trust_env = True
        self._post = post

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        assert self.trust_env is False
        return self._post(url, **kwargs)


def patch_post(monkeypatch, post):
    monkeypatch.setattr(router.requests, "Session", lambda: FakeSession(post))


@pytest.fixture(autouse=True)
def managed_browser_context(monkeypatch):
    monkeypatch.setenv(
        "ZETTLAB_BROWSER_ACTION_URL",
        "http://127.0.0.1:19090/api/v1/internal/browser/action",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "agent-token")
    tokens = set_session_vars(
        platform="zet_agent",
        chat_id="zettlab:alice:agent-1:chat-1",
        session_key="zettlab:alice:agent-1:chat-1",
    )
    yield
    clear_session_vars(tokens)


def test_router_delegates_to_camofox_without_keeping_local_state(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse(200, {
            "success": True,
            "backend": "camofox",
            "delegated": True,
        })

    patch_post(monkeypatch, post)
    route = router.route_browser_action("navigate", {"url": "https://example.com"})
    assert route == router.BrowserRoute("camofox")
    assert seen["headers"]["X-Zettlab-Agent-Action-Token"] == "agent-token"
    assert seen["json"] == {
        "session_id": "zettlab:alice:agent-1:chat-1",
        "action": "navigate",
        "params": {"url": "https://example.com"},
    }


def test_router_returns_desktop_action_result(monkeypatch):
    def post(_url, **_kwargs):
        return FakeResponse(200, {
            "success": True,
            "backend": "desktop",
            "result": {"url": "https://example.com", "title": "Example"},
        })

    patch_post(monkeypatch, post)
    route = router.route_browser_action("navigate", {"url": "https://example.com"})
    assert route.backend == "desktop"
    assert json.loads(route.result) == {
        "success": True,
        "url": "https://example.com",
        "title": "Example",
    }


def test_router_does_not_fallback_after_authoritative_error(monkeypatch):
    def post(_url, **_kwargs):
        return FakeResponse(503, {
            "success": False,
            "backend": "desktop",
            "code": "browser_host_unavailable",
            "error": "not connected",
        })

    patch_post(monkeypatch, post)
    route = router.route_browser_action("snapshot")
    assert route.backend == "error"
    assert json.loads(route.result)["code"] == "browser_host_unavailable"


def test_non_zettlab_session_does_not_call_managed_router(monkeypatch):
    clear_session_vars([])
    tokens = set_session_vars(platform="zet_agent", session_key="api-unowned")
    called = False

    def post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("request must not be sent")

    patch_post(monkeypatch, post)
    assert router.route_browser_action("navigate", {"url": "https://example.com"}) is None
    assert called is False
    clear_session_vars(tokens)


def test_manual_page_snapshot_uses_desktop_without_ai_navigate(monkeypatch):
    actions = []

    def post(_url, **kwargs):
        actions.append(kwargs["json"]["action"])
        return FakeResponse(200, {
            "success": True,
            "backend": "desktop",
            "result": {
                "snapshot": '- button "Continue" [ref=e1]',
                "element_count": 1,
                "url": "https://example.com",
            },
        })

    patch_post(monkeypatch, post)
    result = json.loads(browser_tool.browser_snapshot())
    assert result["success"] is True
    assert result["element_count"] == 1
    assert actions == ["snapshot"]


@pytest.mark.parametrize(
    ("invoke", "action"),
    [
        (lambda: browser_tool.browser_snapshot(), "snapshot"),
        (lambda: browser_tool.browser_click("@e1"), "click"),
        (lambda: browser_tool.browser_type("@e1", "hello"), "type"),
        (lambda: browser_tool.browser_scroll("down"), "scroll"),
        (lambda: browser_tool.browser_back(), "back"),
        (lambda: browser_tool.browser_press("Enter"), "press"),
        (lambda: browser_tool.browser_console(), "console"),
        (lambda: browser_tool.browser_get_images(), "get_images"),
        (lambda: browser_tool.browser_vision("what is shown?"), "vision"),
    ],
)
def test_followup_tools_ask_authoritative_router(monkeypatch, invoke, action):
    calls = []

    def route(actual_action, _params=None):
        calls.append(actual_action)
        return SimpleNamespace(
            backend="error",
            result=json.dumps({"success": False, "code": "bound-backend-error"}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    result = invoke()
    assert json.loads(result)["code"] == "bound-backend-error"
    assert calls == [action]


def test_browser_navigate_executes_camofox_only_after_router_delegates(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda action, params=None: SimpleNamespace(backend="camofox", result=None),
    )
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda _key: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)
    from tools import browser_camofox

    monkeypatch.setattr(
        browser_camofox,
        "camofox_navigate",
        lambda url, task_id=None: json.dumps({"success": True, "url": url}),
    )
    result = json.loads(browser_tool.browser_navigate("https://example.com"))
    assert result == {"success": True, "url": "https://example.com"}


def test_multiplex_config_uses_active_profile_secret_scope(monkeypatch):
    monkeypatch.setenv(
        "ZETTLAB_BROWSER_ACTION_URL",
        "http://127.0.0.1:1/stale-browser-router",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "stale-profile-token")
    scoped_url = "http://127.0.0.1:19090/api/v1/internal/browser/action"
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse(200, {
            "success": True,
            "backend": "desktop",
            "result": {"url": "https://example.com/", "title": "Example"},
        })

    patch_post(monkeypatch, post)
    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "ZETTLAB_BROWSER_ACTION_URL": scoped_url,
        "ZETTLAB_AGENT_ACTION_TOKEN": "active-profile-token",
    })
    try:
        assert router.is_managed_browser_configured() is True
        route = router.route_browser_action("navigate", {"url": "https://example.com"})
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(was_multiplex)

    assert route.backend == "desktop"
    assert seen["url"] == scoped_url
    assert seen["headers"]["X-Zettlab-Agent-Action-Token"] == "active-profile-token"


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/api/v1/internal/browser/action",
        "http://127.0.0.1:not-a-port/api/v1/internal/browser/action",
    ],
)
def test_router_rejects_untrusted_action_endpoint(monkeypatch, url):
    monkeypatch.setenv("ZETTLAB_BROWSER_ACTION_URL", url)
    assert router.is_managed_browser_configured() is False


def test_session_id_falls_back_to_process_env_on_import_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "gateway.session_context", None)
    monkeypatch.setenv("HERMES_SESSION_KEY", "zettlab:alice:agent-1:chat-9")
    assert router._session_id() == "zettlab:alice:agent-1:chat-9"


def test_session_id_falls_back_to_process_env_when_session_context_raises(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("session context unavailable")

    monkeypatch.setattr("gateway.session_context.get_session_env", boom)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setenv("HERMES_SESSION_ID", "zettlab:alice:agent-1:chat-9")
    assert router._session_id() == "zettlab:alice:agent-1:chat-9"


def _deny_local_browser(monkeypatch):
    """Pin every non-desktop availability source to unavailable."""

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("agent-browser not installed")

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: None)
    monkeypatch.setattr(browser_tool, "_find_agent_browser", missing)


def test_check_browser_requirements_probes_host_status(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse(200, {"ok": True, "result": {"host_online": True}})

    patch_post(monkeypatch, post)
    _deny_local_browser(monkeypatch)
    assert browser_tool.check_browser_requirements() is True
    assert seen["json"] == {
        "session_id": "zettlab:alice:agent-1:chat-1",
        "action": "host_status",
        "params": {},
    }
    assert seen["headers"]["X-Zettlab-Agent-Action-Token"] == "agent-token"
    assert seen["timeout"] <= 2


def _probe_network_error(_url, **_kwargs):
    raise requests.RequestException("router unreachable")


@pytest.mark.parametrize(
    "post",
    [
        pytest.param(
            lambda _url, **_kwargs: FakeResponse(
                200, {"ok": True, "result": {"host_online": False}}
            ),
            id="host-offline",
        ),
        pytest.param(_probe_network_error, id="probe-network-error"),
        pytest.param(
            lambda _url, **_kwargs: FakeResponse(
                404, {"success": False, "error": "unknown action: host_status"}
            ),
            id="legacy-ls-unknown-action",
        ),
        pytest.param(
            lambda _url, **_kwargs: FakeResponse(200, {"success": True, "backend": "desktop"}),
            id="legacy-ls-no-ok-envelope",
        ),
    ],
)
def test_check_browser_requirements_falls_back_without_live_host(monkeypatch, post):
    patch_post(monkeypatch, post)
    _deny_local_browser(monkeypatch)
    # Desktop configuration alone must not advertise the browser tools…
    assert browser_tool.check_browser_requirements() is False
    # …while the existing Camofox check still applies unchanged.
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
    assert browser_tool.check_browser_requirements() is True
