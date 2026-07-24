"""Flow tests for the epoch-based takeover/handback synchronization.

The local-server proxy keeps one monotonic epoch per tab that increments when
human control ends. Hermes declares its last-seen epoch on tab operations; a
mismatch returns 409 browser_epoch_stale and recovery is a single stateless
step: adopt the new epoch, take one privacy-filtered snapshot, retry.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.browser_camofox import (
    _EPOCH_HEADER,
    _adopt_epoch_from_response,
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


def _epoch_stale(epoch: int = 3) -> requests.HTTPError:
    return _http_error(
        409,
        {
            "error": "browser_epoch_stale",
            "message": "page state changed after human control; snapshot the tab before continuing",
            "epoch": epoch,
        },
    )


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
        "privacy_filter_after_handback": False,
        "epoch": 2,
    }


def test_epoch_stale_resnapshots_and_returns_retryable(managed_session):
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", return_value={
            "snapshot": '- button "Continue" [e9]',
            "refsCount": 1,
        }) as mock_get,
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert result["success"] is False
    assert result["error"] == "browser_epoch_stale"
    assert result["retryable"] is True
    assert result["resnapshot_completed"] is True
    assert result["snapshot"] == '- button "Continue" [e9]'
    assert result["element_count"] == 1
    # The session adopted the new epoch and enabled the privacy filter.
    assert managed_session["epoch"] == 3
    assert managed_session["privacy_filter_after_handback"] is True
    assert mock_get.call_args.args == ("/tabs/tab-agent/snapshot",)
    assert mock_get.call_args.kwargs["params"] == {"userId": "hermes_profile"}
    assert mock_get.call_args.kwargs["session"] is managed_session


def test_epoch_stale_without_session_context_is_still_retryable():
    result_json = None
    with patch("tools.browser_camofox._get_session", return_value={"user_id": "u", "tab_id": None}):
        with patch("tools.browser_camofox._post"):
            from tools.browser_camofox import _retryable_control_result

            result_json = _retryable_control_result(_epoch_stale(9), None)

    result = json.loads(result_json)
    assert result["error"] == "browser_epoch_stale"
    assert result["retryable"] is True
    assert result["resnapshot_completed"] is False


def test_epoch_stale_snapshot_failure_stays_retryable(managed_session):
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", side_effect=requests.ConnectionError("boom")),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    # No auto-close, no dead end: the model can just retry.
    assert result["error"] == "browser_epoch_stale"
    assert result["retryable"] is True
    assert result["resnapshot_completed"] is False
    assert managed_session["privacy_filter_after_handback"] is True
    assert managed_session["epoch"] == 3


def test_epoch_stale_redacts_sensitive_human_page_state(managed_session):
    sensitive_snapshot = (
        '- textbox "Password": hunter2\n'
        '- textbox "Enter the 6-digit code": 654321\n'
        '- textbox "Card number": 4242424242424242\n'
        '- textbox "Nickname": private nickname\n'
        '- heading "eyJabcdefghijk.payload.signature"'
    )
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", return_value={
            "snapshot": sensitive_snapshot,
            "refsCount": 4,
        }),
    ):
        result = json.loads(camofox_click("@e4", task_id="agent-task"))

    encoded = json.dumps(result)
    assert "hunter2" not in encoded
    assert "654321" not in encoded
    assert "4242424242424242" not in encoded
    assert "private nickname" not in encoded
    assert "eyJabcdefghijk" not in encoded
    assert result["resnapshot_completed"] is True


def test_privacy_filter_blocks_raw_vision_and_filters_snapshot(managed_session):
    private_snapshot = 'textbox "Nickname": private nickname\n- button "Continue" [e9]'

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", return_value={
            "snapshot": private_snapshot,
            "refsCount": 1,
        }),
        patch("tools.browser_camofox._get_raw") as mock_get_raw,
        patch("agent.auxiliary_client.call_llm") as mock_llm,
    ):
        handback = json.loads(camofox_click("@e4", task_id="agent-task"))
        later_snapshot = json.loads(camofox_snapshot(task_id="agent-task"))
        vision = json.loads(camofox_vision("What is visible?", annotate=True, task_id="agent-task"))

    assert handback["resnapshot_completed"] is True
    assert managed_session["privacy_filter_after_handback"] is True
    assert "private nickname" not in later_snapshot["snapshot"]
    assert "[REDACTED sensitive form control]" in later_snapshot["snapshot"]
    assert vision["success"] is False
    assert "blocked after human control" in vision["error"]
    mock_get_raw.assert_not_called()
    mock_llm.assert_not_called()


def test_epoch_header_change_on_success_enables_privacy_filter(managed_session):
    """A snapshot taken after handback without any 409 must still filter.

    Snapshots are always admitted, so the first post-handback call can be a
    plain browser_snapshot. The changed epoch header on that response is the
    only staleness signal, and it must flip the privacy filter on.
    """
    response = MagicMock()
    response.headers = {_EPOCH_HEADER: "3"}
    _adopt_epoch_from_response(managed_session, response)

    assert managed_session["epoch"] == 3
    assert managed_session["privacy_filter_after_handback"] is True


def test_epoch_header_same_value_keeps_filter_off(managed_session):
    response = MagicMock()
    response.headers = {_EPOCH_HEADER: "2"}
    _adopt_epoch_from_response(managed_session, response)

    assert managed_session["epoch"] == 2
    assert managed_session["privacy_filter_after_handback"] is False


def test_first_epoch_sighting_does_not_enable_filter(managed_session):
    managed_session["epoch"] = None
    response = MagicMock()
    response.headers = {_EPOCH_HEADER: "5"}
    _adopt_epoch_from_response(managed_session, response)

    assert managed_session["epoch"] == 5
    assert managed_session["privacy_filter_after_handback"] is False


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


def test_console_evaluate_uses_epoch_recovery(managed_session):
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", return_value={
            "snapshot": '- heading "Signed in"',
            "refsCount": 0,
        }),
    ):
        result = json.loads(_camofox_eval("document.title", "agent-task"))

    assert result["error"] == "browser_epoch_stale"
    assert result["retryable"] is True
    assert result["resnapshot_completed"] is True


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
        patch("tools.browser_camofox.camofox_scroll", return_value=json.dumps({"success": False, "error": "browser_epoch_stale"})) as scroll,
    ):
        result = json.loads(browser_scroll("down", "agent-task"))

    assert result["error"] == "browser_epoch_stale"
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
