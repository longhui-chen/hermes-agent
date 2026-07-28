import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import browser_desktop_host as desktop, browser_tool


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
    monkeypatch.setattr(desktop.requests, "Session", lambda: FakeSession(post))


@pytest.fixture(autouse=True)
def desktop_context(monkeypatch):
    monkeypatch.setenv(
        "ZETTLAB_DESKTOP_BROWSER_HOST_URL",
        "http://127.0.0.1:19090/api/v1/internal/browser-host/action",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "agent-token")
    tokens = set_session_vars(
        platform="zet_agent",
        chat_id="zettlab:alice:agent-1:chat-1",
        session_key="zettlab:alice:agent-1:chat-1",
    )
    desktop._reset_desktop_browser_state_for_tests()
    yield
    desktop._reset_desktop_browser_state_for_tests()
    clear_session_vars(tokens)


def test_navigate_falls_through_when_no_pc_host(monkeypatch):
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse(503, {
            "success": False,
            "code": "browser_host_unavailable",
            "error": "not connected",
        })

    patch_post(monkeypatch, post)
    assert desktop.desktop_browser_navigate("https://example.com") is None
    assert desktop.has_desktop_browser_session() is False
    assert seen["headers"]["X-Zettlab-Agent-Action-Token"] == "agent-token"
    assert seen["json"] == {
        "session_id": "zettlab:alice:agent-1:chat-1",
        "action": "navigate",
        "params": {"url": "https://example.com"},
    }


def test_successful_navigate_pins_followup_actions_to_desktop(monkeypatch):
    calls = []

    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        if kwargs["json"]["action"] == "navigate":
            return FakeResponse(200, {
                "success": True,
                "result": {"url": "https://example.com", "title": "Example"},
            })
        return FakeResponse(200, {
            "success": True,
            "result": {"snapshot": '- button "Continue" [ref=e1]', "element_count": 1},
        })

    patch_post(monkeypatch, post)
    result = json.loads(desktop.desktop_browser_navigate("https://example.com"))
    assert result == {
        "success": True,
        "url": "https://example.com",
        "title": "Example",
    }
    assert desktop.has_desktop_browser_session() is True

    snapshot = json.loads(desktop.desktop_browser_snapshot())
    assert snapshot["success"] is True
    assert snapshot["element_count"] == 1
    assert calls[-1]["session_id"] == "zettlab:alice:agent-1:chat-1"
    assert calls[-1]["action"] == "snapshot"


def test_non_zettlab_session_cannot_use_desktop_host(monkeypatch):
    clear_session_vars([])
    tokens = set_session_vars(platform="zet_agent", session_key="api-unowned")
    called = False

    def post(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("request must not be sent")

    patch_post(monkeypatch, post)
    assert desktop.desktop_browser_navigate("https://example.com") is None
    assert called is False
    clear_session_vars(tokens)


def test_browser_tools_route_a_pinned_session_through_desktop(monkeypatch):
    actions = []

    def post(_url, **kwargs):
        action = kwargs["json"]["action"]
        actions.append(action)
        result = (
            {"url": "https://example.com", "title": "Example"}
            if action == "navigate"
            else {"snapshot": '- button "Continue" [ref=e1]', "element_count": 1}
        )
        return FakeResponse(200, {"success": True, "result": result})

    patch_post(monkeypatch, post)
    monkeypatch.setattr(browser_tool, "_is_desktop_browser_configured", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_is_local_sidecar_key", lambda _key: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda _url: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda _url: None)

    navigation = json.loads(browser_tool.browser_navigate("https://example.com"))
    assert navigation["success"] is True
    snapshot = json.loads(browser_tool.browser_snapshot())
    assert snapshot["success"] is True
    assert actions == ["navigate", "snapshot"]
