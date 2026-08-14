import json
import sys
from types import SimpleNamespace

import pytest
import requests

from agent import secret_scope
from gateway.session_context import (
    clear_session_vars,
    pop_zettlab_browser_session_token,
    push_zettlab_browser_session_token,
    set_session_vars,
)
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
    browser_token = push_zettlab_browser_session_token("browser-scope-token")
    yield
    pop_zettlab_browser_session_token(browser_token)
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
    assert seen["headers"]["X-Zettlab-Browser-Session-Token"] == "browser-scope-token"
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
        (lambda: browser_tool.browser_vision("what is shown?"), "screenshot"),
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
    assert seen["headers"]["X-Zettlab-Browser-Session-Token"] == "browser-scope-token"


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
    assert seen["headers"]["X-Zettlab-Browser-Session-Token"] == "browser-scope-token"
    assert seen["timeout"] <= 2


def test_router_fails_closed_without_request_scope_token(monkeypatch):
    called = False

    def post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("request must not be sent")

    patch_post(monkeypatch, post)
    reset = push_zettlab_browser_session_token("")
    try:
        assert router.is_managed_browser_configured() is False
        assert router.route_browser_action("snapshot") is None
    finally:
        pop_zettlab_browser_session_token(reset)
    assert called is False


def test_managed_navigate_blocks_private_redirect_and_closes_page(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "navigate":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({
                    "success": True,
                    "url": "http://127.0.0.1/admin",
                    "title": "Internal",
                }),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "closed": True}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda _key: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(
        browser_tool,
        "_is_safe_url",
        lambda candidate: candidate == "https://example.com/redirect",
    )
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)

    result = json.loads(browser_tool.browser_navigate("https://example.com/redirect"))
    assert result["success"] is False
    assert "private or internal" in result["error"]
    assert actions == ["navigate", "close"]


def test_managed_navigate_allows_lan_page_for_local_backend(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "url": "http://192.168.1.10/ui",
                "title": "Device Web UI",
            }),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda _key: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)

    result = json.loads(browser_tool.browser_navigate("http://192.168.1.10/ui"))
    assert result["success"] is True
    assert result["url"] == "http://192.168.1.10/ui"
    assert actions == ["navigate"]


def test_managed_navigate_still_blocks_metadata_for_local_backend(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "navigate":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({
                    "success": True,
                    "url": "http://169.254.169.254/latest/meta-data/",
                    "title": "",
                }),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "closed": True}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda _key: False)
    monkeypatch.setattr(
        browser_tool,
        "_is_always_blocked_url",
        lambda url: "169.254.169.254" in url,
    )
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: True)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)

    result = json.loads(browser_tool.browser_navigate("https://example.com/"))
    assert result["success"] is False
    assert "metadata" in result["error"]
    assert actions == ["navigate", "close"]


@pytest.mark.parametrize(
    "url",
    ["about:blank", "about:srcdoc", "about:blank#reset", "ABOUT:BLANK"],
)
def test_managed_page_safety_skips_browser_internal_blank_pages(monkeypatch, url):
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    assert browser_tool._managed_page_safety_error(url) is None


@pytest.mark.parametrize(
    "url",
    ["about:config", "aboutx:blank", "http://about.blank/"],
)
def test_managed_page_safety_does_not_exempt_non_blank_urls(monkeypatch, url):
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: False)
    error = browser_tool._managed_page_safety_error(url)
    assert error is not None
    assert "private or internal" in error


def test_managed_snapshot_allows_about_blank_without_closing(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "snapshot": "",
                "element_count": 0,
                "url": "about:blank",
            }),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    # Real _is_safe_url / _is_always_blocked_url on purpose: about:blank has no
    # http(s) scheme, so only the blank-page exemption can let it through.
    result = json.loads(browser_tool.browser_snapshot())
    assert result["success"] is True
    assert actions == ["snapshot"]


def test_managed_console_eval_never_sends_expression_and_rejects_desktop(monkeypatch):
    seen = []

    def route(action, params=None):
        seen.append((action, params))
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "result": "should never be used"}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    result = json.loads(browser_tool.browser_console(expression="document.title"))
    assert result["success"] is False
    assert result["code"] == "browser_eval_not_supported_on_managed_desktop"
    assert seen == [("console", {"clear": False})]


def test_managed_console_eval_surfaces_router_error_without_local_eval(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="error",
            result=json.dumps({
                "success": False,
                "code": "browser_action_not_supported_by_backend",
            }),
        ),
    )

    def never_eval(*_args, **_kwargs):
        raise AssertionError("local eval must not run after a terminal router error")

    monkeypatch.setattr(browser_tool, "_browser_eval", never_eval)
    result = json.loads(browser_tool.browser_console(expression="1+1"))
    assert result["success"] is False
    assert result["code"] == "browser_action_not_supported_by_backend"


def test_managed_console_eval_camofox_delegation_uses_local_eval_path(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(backend="camofox", result=None),
    )
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(
        browser_tool,
        "_browser_eval",
        lambda expression, task_id=None: json.dumps({"success": True, "result": "2"}),
    )
    result = json.loads(browser_tool.browser_console(expression="1+1"))
    assert result == {"success": True, "result": "2"}


def test_managed_console_output_redacts_desktop_payload(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "console_messages": [{"type": "log", "text": "token sk-secret"}],
                "url": "https://example.com",
            }),
        ),
    )
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: True)
    monkeypatch.setattr(
        browser_tool,
        "_redact_browser_output",
        lambda value: {**value, "console_messages": "redacted"}
        if isinstance(value, dict)
        else value,
    )

    result = json.loads(browser_tool.browser_console())
    assert result["success"] is True
    assert result["console_messages"] == "redacted"


def test_managed_console_output_blocks_private_page_and_closes(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "console":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({
                    "success": True,
                    "console_messages": [{"type": "log", "text": "internal secret"}],
                    "url": "http://127.0.0.1:8080/internal",
                }),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "closed": True}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: False)

    result = json.loads(browser_tool.browser_console())
    assert result["success"] is False
    assert "private or internal" in result["error"]
    assert "internal secret" not in json.dumps(result)
    assert actions == ["console", "close"]


def test_managed_back_checks_landed_url_and_closes_private_page(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "back":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({
                    "success": True,
                    "url": "http://169.254.169.254/latest/meta-data/",
                }),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "closed": True}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(
        browser_tool,
        "_managed_page_safety_error",
        lambda url: "Blocked private page" if url.startswith("http://169.254.") else None,
    )

    result = json.loads(browser_tool.browser_back())
    assert result == {"success": False, "error": "Blocked private page"}
    assert actions == ["back", "close"]


def test_managed_back_preserves_public_result_schema(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "url": "https://example.com/previous",
                "title": "Previous",
            }),
        ),
    )
    monkeypatch.setattr(browser_tool, "_managed_page_safety_error", lambda _url: None)
    monkeypatch.setattr(browser_tool, "_redact_browser_output", lambda value: value)

    result = json.loads(browser_tool.browser_back())
    assert result == {"success": True, "url": "https://example.com/previous"}


@pytest.mark.parametrize("tool_call", [browser_tool.browser_back, browser_tool.browser_get_images])
def test_managed_content_action_fails_closed_without_page_url(monkeypatch, tool_call):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "close":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({"success": True, "closed": True}),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "images": [{"src": "https://example.com/secret.png"}],
            }),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)

    result = json.loads(tool_call())
    assert result["success"] is False
    assert result["code"] == "invalid_browser_router_response"
    assert "secret.png" not in json.dumps(result)
    assert actions[-1] == "close"


def test_managed_get_images_checks_url_and_redacts_result(monkeypatch):
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "url": "https://example.com/gallery",
                "images": [
                    {
                        "src": "https://example.com/image.png?token=secret",
                        "alt": "secret alt",
                        "width": 640,
                        "height": 480,
                    },
                    {"src": "data:image/png;base64,secret", "alt": "inline"},
                ],
                "count": 2,
            }),
        ),
    )
    monkeypatch.setattr(browser_tool, "_managed_page_safety_error", lambda _url: None)
    monkeypatch.setattr(
        browser_tool,
        "_redact_browser_output",
        lambda value: [{**value[0], "src": "redacted"}]
        if isinstance(value, list)
        else value,
    )

    result = json.loads(browser_tool.browser_get_images())
    assert result == {
        "success": True,
        "images": [{
            "src": "redacted",
            "alt": "secret alt",
            "width": 640,
            "height": 480,
        }],
        "count": 1,
    }


def test_managed_get_images_does_not_return_private_page_content(monkeypatch):
    actions = []

    def route(action, _params=None):
        actions.append(action)
        if action == "get_images":
            return SimpleNamespace(
                backend="desktop",
                result=json.dumps({
                    "success": True,
                    "url": "http://127.0.0.1/internal",
                    "images": [{"src": "http://127.0.0.1/secret.png"}],
                }),
            )
        return SimpleNamespace(
            backend="desktop",
            result=json.dumps({"success": True, "closed": True}),
        )

    monkeypatch.setattr(browser_tool, "_route_browser_action", route)
    monkeypatch.setattr(
        browser_tool,
        "_managed_page_safety_error",
        lambda url: "Blocked private page" if url.startswith("http://127.") else None,
    )

    result = json.loads(browser_tool.browser_get_images())
    assert result == {"success": False, "error": "Blocked private page"}
    assert "secret.png" not in json.dumps(result)
    assert actions == ["get_images", "close"]


def test_managed_snapshot_truncates_and_redacts_before_return(monkeypatch):
    monkeypatch.setattr(browser_tool, "SNAPSHOT_SUMMARIZE_THRESHOLD", 8)
    monkeypatch.setattr(browser_tool, "_truncate_snapshot", lambda _value: "trimmed-secret")
    monkeypatch.setattr(
        browser_tool,
        "_redact_browser_output",
        lambda value: {
            **value,
            "snapshot": "redacted",
        } if isinstance(value, dict) else value,
    )
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "snapshot": "a very long snapshot",
                "element_count": 3,
                "url": "https://example.com",
            }),
        ),
    )
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: True)

    result = json.loads(browser_tool.browser_snapshot())
    assert result["snapshot"] == "redacted"
    assert result["element_count"] == 3


def test_managed_vision_uses_native_multimodal_pipeline(monkeypatch, tmp_path):
    import base64
    from tools import vision_tools

    png = b"\x89PNG\r\n\x1a\nmanaged"
    monkeypatch.setattr(
        browser_tool,
        "_route_browser_action",
        lambda _action, _params=None: SimpleNamespace(
            backend="desktop",
            result=json.dumps({
                "success": True,
                "data": base64.b64encode(png).decode("ascii"),
                "mime_type": "image/png",
                "url": "https://example.com",
            }),
        ),
    )
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda _url: True)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "chromium")
    monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
    monkeypatch.setattr(
        vision_tools,
        "_build_native_vision_tool_result",
        lambda **kwargs: {
            "type": "multimodal",
            "text_summary": kwargs["question"],
            "meta": {"image_size_bytes": kwargs["image_size_bytes"]},
        },
    )

    result = browser_tool.browser_vision("what is shown?")
    assert isinstance(result, dict)
    assert result["type"] == "multimodal"
    assert result["meta"]["image_size_bytes"] == len(png)
    assert result["meta"]["screenshot_path"].endswith(".png")


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
