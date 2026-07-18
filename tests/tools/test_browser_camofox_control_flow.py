"""Flow tests for local-server browser takeover and handback recovery."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.browser_camofox import (
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_snapshot,
    camofox_vision,
)
from tools.browser_tool import _camofox_eval, browser_scroll


def _http_error(status: int, payload: dict) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    return requests.HTTPError(response=response)


def _pending_tabs(origin: str = "https://example.com", url: str = "") -> dict:
    return {
        "tabs": [{
            "tabId": "tab-agent",
            "listItemId": "task_opaque",
            "url": url,
            "title": "",
            "resumeUrlOrigin": origin,
        }],
    }


def _current_tabs(url: str = "https://example.com/account") -> dict:
    return {
        "tabs": [{
            "tabId": "tab-agent",
            "listItemId": "task_opaque",
            "url": url,
            "title": "Account",
        }],
    }


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
            _pending_tabs(),
            {"snapshot": '- button "Continue" [e9]', "refsCount": 1},
            _current_tabs(),
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
    assert mock_get.call_args_list[0].args == ("/tabs",)
    assert mock_get.call_args_list[1].args == ("/tabs/tab-agent/snapshot",)
    assert mock_get.call_args_list[1].kwargs == {"params": {"userId": "hermes_profile"}}
    assert mock_get.call_args_list[2].args == ("/tabs",)
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
            _pending_tabs(),
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            _current_tabs("https://example.com"),
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is False
    assert mock_post.call_count == 1


def test_handback_ack_does_not_require_tab_title(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=[stale, {"ok": True}]) as mock_post,
        patch("tools.browser_camofox._get", side_effect=[
            _pending_tabs(),
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            _current_tabs(),
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is True
    assert result["title"] == "[REDACTED after human control]"
    assert mock_post.call_count == 2


def test_handback_redacts_sensitive_human_page_state(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=[stale, {"ok": True}]),
        patch("tools.browser_camofox._get", side_effect=[
            _pending_tabs("https://example.com:8443"),
            {
                "snapshot": '- textbox "Password": hunter2\n- textbox "Enter the 6-digit code": 654321\n- textbox "Card number": 4242424242424242\n- textbox "Nickname": private nickname\ntextbox "Email": root@example.com\nsearchbox "People": Alice Root\n- textbox: Alice Smith\n- spinbutton "Amount": 1200\n- combobox "Account":\n  - option "Checking 1234" [selected]\n- listbox "Address":\n  - option "1 Private Lane" [selected]\n- heading "eyJabcdefghijk.payload.signature"',
                "refsCount": 4,
            },
            _current_tabs(
                "https://alice:password@example.com:8443/reset/path-secret?session_token=raw-secret&redirect=https%3A%2F%2Fnested.example%2F%3FaccessToken%3Dnested-secret#authorization_code=raw-fragment-secret"
            ),
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    encoded = json.dumps(result)
    assert "hunter2" not in encoded
    assert "654321" not in encoded
    assert "4242424242424242" not in encoded
    assert "private nickname" not in encoded
    assert "root@example.com" not in encoded
    assert "Alice Root" not in encoded
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


def test_handback_privacy_filter_blocks_raw_vision_and_filters_snapshot(
    managed_session,
):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    private_snapshot = 'textbox "Nickname": private nickname\n- button "Continue" [e9]'

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=[stale, {"ok": True}]),
        patch("tools.browser_camofox._get", side_effect=[
            _pending_tabs(),
            {"snapshot": private_snapshot, "refsCount": 1},
            _current_tabs(),
            {"snapshot": private_snapshot, "refsCount": 1},
        ]),
        patch("tools.browser_camofox._get_raw") as mock_get_raw,
        patch("agent.auxiliary_client.call_llm") as mock_llm,
    ):
        handback = json.loads(camofox_click("@e4", task_id="agent-task"))
        later_snapshot = json.loads(camofox_snapshot(task_id="agent-task"))
        vision = json.loads(camofox_vision("What is visible?", annotate=True, task_id="agent-task"))

    assert handback["resume_acknowledged"] is True
    assert managed_session["privacy_filter_after_handback"] is True
    assert "private nickname" not in later_snapshot["snapshot"]
    assert "[REDACTED sensitive form control]" in later_snapshot["snapshot"]
    assert vision["success"] is False
    assert "blocked after human control" in vision["error"]
    mock_get_raw.assert_not_called()
    mock_llm.assert_not_called()


@pytest.mark.parametrize(
    "current_url",
    [
        "http://localhost/admin",
        "http://10.0.0.8/private",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_handback_private_or_metadata_url_stays_blocked(managed_session, current_url):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", return_value=_pending_tabs(current_url)) as mock_get,
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["error"] == "browser_handback_origin_blocked"
    assert result["retryable"] is False
    assert result["resnapshot_completed"] is False
    assert result["resume_acknowledged"] is False
    assert current_url not in json.dumps(result)
    assert managed_session["privacy_filter_after_handback"] is True
    mock_post.assert_called_once()
    mock_get.assert_called_once_with("/tabs", params={"userId": "hermes_profile"}, timeout=5)


@pytest.mark.parametrize(
    ("is_safe", "is_private"),
    [
        (False, False),  # Hermes DNS failure while the browser may still resolve it.
        (True, True),  # Private split-DNS answer despite an opted-out safe-url policy.
    ],
)
def test_handback_origin_fails_closed_on_dns_or_private_resolution(
    managed_session,
    is_safe,
    is_private,
):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", return_value=_pending_tabs("https://split.example")) as mock_get,
        patch("tools.browser_tool._is_always_blocked_url", return_value=False),
        patch("tools.browser_tool._is_safe_url", return_value=is_safe),
        patch("tools.browser_tool._url_is_private", return_value=is_private),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["error"] == "browser_handback_origin_blocked"
    assert result["resnapshot_completed"] is False
    assert result["resume_acknowledged"] is False
    assert "snapshot" not in result
    mock_post.assert_called_once()
    mock_get.assert_called_once()


def test_handback_rejects_nonempty_pending_url_even_with_safe_origin(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch(
            "tools.browser_camofox._get",
            return_value=_pending_tabs(
                "https://example.com",
                url="https://example.com/hidden-path?secret=value",
            ),
        ) as mock_get,
        patch("tools.browser_camofox._unsafe_handback_url", return_value=False),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["error"] == "browser_handback_origin_blocked"
    assert result["resnapshot_completed"] is False
    assert result["resume_acknowledged"] is False
    assert "snapshot" not in result
    assert "hidden-path" not in json.dumps(result)
    mock_post.assert_called_once()
    mock_get.assert_called_once()


def test_handback_does_not_expose_snapshot_or_ack_if_url_turns_unsafe(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    private_snapshot = '- heading "Internal account data"\n- textbox "Secret": value'
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", side_effect=[
            _pending_tabs(),
            {"snapshot": private_snapshot, "refsCount": 1},
            _current_tabs("http://127.0.0.1/private"),
        ]),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["error"] == "browser_handback_url_blocked"
    assert result["retryable"] is False
    assert result["resnapshot_completed"] is False
    assert result["resume_acknowledged"] is False
    assert "snapshot" not in result
    assert "Internal account data" not in json.dumps(result)
    assert managed_session["privacy_filter_after_handback"] is True
    mock_post.assert_called_once()


def test_agent_navigation_clears_handback_privacy_filter(managed_session):
    managed_session["privacy_filter_after_handback"] = True
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/next"}),
        patch("tools.browser_camofox._get", return_value={"snapshot": '- heading "Next"', "refsCount": 0}),
    ):
        result = json.loads(camofox_navigate("https://example.com/next", task_id="agent-task"))

    assert result["success"] is True
    assert managed_session["privacy_filter_after_handback"] is False


def test_handback_does_not_ack_without_current_tab_metadata(managed_session):
    stale = _http_error(409, {"error": "browser_resnapshot_required", "resume_token": "resume-secret"})
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=stale) as mock_post,
        patch("tools.browser_camofox._get", return_value={"tabs": []}),
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
            _pending_tabs(),
            {"snapshot": '- heading "Signed in"', "refsCount": 0},
            _current_tabs("https://example.com"),
        ]),
    ):
        result = json.loads(_camofox_eval("document.title", "agent-task"))

    assert result["error"] == "browser_resnapshot_required"
    assert result["resnapshot_completed"] is True
    assert result["resume_acknowledged"] is True


def test_console_evaluate_is_blocked_while_handback_privacy_filter_is_active(managed_session):
    managed_session["privacy_filter_after_handback"] = True
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=managed_session),
        patch("tools.browser_camofox._post") as mock_post,
    ):
        result = json.loads(_camofox_eval("document.querySelector('input').value", "agent-task"))

    assert result["success"] is False
    assert "blocked after human control" in result["error"]
    mock_post.assert_not_called()


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
