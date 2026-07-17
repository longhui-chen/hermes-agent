"""Flow tests for local-server browser takeover and handback recovery."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.browser_camofox import camofox_click, camofox_close, camofox_navigate
from tools.browser_tool import _camofox_eval, browser_scroll


def _http_error(status: int, payload: dict) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    return requests.HTTPError(response=response)


@pytest.fixture
def managed_session(monkeypatch):
    monkeypatch.setenv("CAMOFOX_URL", "http://127.0.0.1:8080/api/v1/internal/browser/camofox")
    monkeypatch.setenv("CAMOFOX_AUTH_MODE", "zettlab_action_token")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    monkeypatch.setenv("CAMOFOX_MANAGED_BY_LOCAL_SERVER", "true")
    return {
        "user_id": "hermes_profile",
        "tab_id": "tab-agent",
        "session_key": "task_opaque",
        "managed": True,
        "adopt_existing_tab": True,
    }


def test_human_controlled_action_returns_retryable_state(managed_session):
    controlled = _http_error(
        423,
        {
            "error": "browser_human_controlled",
            "message": "The user is controlling this tab.",
            "retry_after_ms": 750,
            "takeover_session_id": "takeover-1",
        },
    )

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=controlled),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result == {
        "success": False,
        "error": "browser_human_controlled",
        "message": "The user is controlling this tab.",
        "retryable": True,
        "retry_after_ms": 750,
        "takeover_session_id": "takeover-1",
    }


def test_handback_resnapshots_then_acks_before_retry(managed_session):
    stale = _http_error(
        409,
        {
            "error": "browser_resnapshot_required",
            "message": "Browser state changed during takeover.",
            "resume_token": "resume-secret",
        },
    )
    post_calls = []

    def fake_post(path, body, timeout=None):
        post_calls.append((path, body, timeout))
        if path.endswith("/click"):
            raise stale
        return {"ok": True}

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=fake_post),
        patch("tools.browser_camofox._get", side_effect=[
            {"snapshot": '- button "Continue" [e9]', "refsCount": 1},
            {"tabs": [{"tabId": "tab-agent", "url": "https://example.com/account", "title": "Account"}]},
        ]) as mock_get,
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["success"] is False
    assert result["error"] == "browser_resnapshot_required"
    assert result["retryable"] is True
    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is True
    assert result["snapshot"] == '- button "Continue" [e9]'
    assert result["url"] == "https://example.com"
    assert result["title"] == "[REDACTED after human control]"
    assert "resume-secret" not in json.dumps(result)
    assert mock_get.call_args_list[0].args == ("/tabs/tab-agent/snapshot",)
    assert mock_get.call_args_list[0].kwargs == {"params": {"userId": "hermes_profile"}}
    assert post_calls[-1] == (
        "/_zettlab/control/resume/ack",
        {
            "userId": "hermes_profile",
            "tabId": "tab-agent",
            "resumeToken": "resume-secret",
        },
        None,
    )


def test_handback_without_resume_token_does_not_ack(managed_session):
    stale = _http_error(
        409,
        {"error": "browser_resnapshot_required", "message": "Resnapshot first."},
    )

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", side_effect=[
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            {"tabs": [{"tabId": "tab-agent", "url": "https://example.com", "title": "Example"}]},
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is False
    assert mock_post.call_count == 1


def test_handback_redacts_sensitive_human_page_state(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=[stale, {"ok": True}]),
        patch("tools.browser_camofox._get", side_effect=[
            {
                "snapshot": '- textbox "Password": hunter2\n- textbox "Enter the 6-digit code": 654321\n- textbox "Card number": 4242424242424242\n- textbox "Nickname": private nickname\n- textbox "Email": alice@example.com\n- textbox: Alice Smith\n- spinbutton "Amount": 1200\n- combobox "Account":\n  - option "Checking 1234" [selected]\n- listbox "Address":\n  - option "1 Private Lane" [selected]\n- heading "eyJabcdefghijk.payload.signature"',
                "refsCount": 4,
            },
            {
                "tabs": [{
                    "tabId": "tab-agent",
                    "url": "https://alice:password@example.com:8443/reset/path-secret?session_token=raw-secret&redirect=https%3A%2F%2Fnested.example%2F%3FaccessToken%3Dnested-secret#authorization_code=raw-fragment-secret",
                    "title": "Alice Smith alice@example.com +86 13800138000 OTP 123456",
                }],
            },
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    encoded = json.dumps(result)
    assert "hunter2" not in encoded
    assert "654321" not in encoded
    assert "4242424242424242" not in encoded
    assert "private nickname" not in encoded
    assert "alice@example.com" not in encoded
    assert "Alice Smith" not in encoded
    assert "1200" not in encoded
    assert "Checking 1234" not in encoded
    assert "1 Private Lane" not in encoded
    assert "eyJabcdefghijk" not in encoded
    assert "raw-secret" not in encoded
    assert "path-secret" not in encoded
    assert "alice" not in encoded
    assert "password" not in encoded
    assert "nested-secret" not in encoded
    assert "raw-fragment-secret" not in encoded
    assert "13800138000" not in encoded
    assert result["url"] == "https://example.com:8443"
    assert "123456" not in encoded
    assert "sk-12345678901234567890" not in encoded
    assert result["resume_acknowledged"] is True


def test_handback_does_not_ack_without_current_tab_metadata(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", side_effect=[
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            {"tabs": []},
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["resnapshot_completed"] is False
    assert result["resume_acknowledged"] is False
    assert mock_post.call_count == 1


def test_console_evaluate_uses_handback_recovery(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=[stale, {"ok": True}]),
        patch("tools.browser_camofox._get", side_effect=[
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            {"tabs": [{"tabId": "tab-agent", "url": "https://example.com", "title": "Example"}]},
        ]),
    ):
        result = json.loads(_camofox_eval("document.title", "agent-task"))

    assert result["error"] == "browser_resnapshot_required"
    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is True


def test_scroll_stops_after_first_control_block():
    with (
        patch("tools.browser_tool._is_camofox_mode", return_value=True),
        patch("tools.browser_camofox.camofox_scroll", return_value=json.dumps({"success": False, "error": "browser_human_controlled"})) as scroll,
    ):
        result = json.loads(browser_scroll("down", "agent-task"))

    assert result["error"] == "browser_human_controlled"
    scroll.assert_called_once()


def test_managed_navigation_emits_exact_takeover_hint(managed_session, monkeypatch):
    monkeypatch.setenv("ZET_AGENT_ID", "agent-42")
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com"}),
        patch("tools.browser_camofox._get", return_value={"snapshot": "", "refsCount": 0}),
    ):
        result = json.loads(camofox_navigate("https://example.com", task_id="agent-task"))

    assert result["ui_hint"] == {
        "type": "takeover_browser",
        "agent_id": "agent-42",
        "browser_session_id": "task_opaque",
        "tab_id": "tab-agent",
    }


def test_managed_close_releases_lease_without_destroying_profile(managed_session):
    with (
        patch("tools.browser_camofox._drop_session", return_value=managed_session),
        patch("tools.browser_camofox._post", return_value={"ok": True}) as mock_post,
        patch("tools.browser_camofox._delete") as mock_delete,
    ):
        result = json.loads(camofox_close("agent-task"))

    assert result == {"success": True, "closed": False, "released": True}
    mock_post.assert_called_once_with("/_zettlab/release", {}, timeout=5)
    mock_delete.assert_not_called()
