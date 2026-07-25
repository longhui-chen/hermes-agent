"""Flow tests for the epoch-based takeover/handback synchronization.

The local-server proxy keeps one monotonic epoch per tab that increments when
human control ends. Hermes declares its last-seen epoch on tab operations; a
mismatch returns 409 browser_epoch_stale and recovery is a single stateless
step: adopt the new epoch, take one privacy-filtered snapshot, retry.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.browser_camofox import (
    _adopt_existing_tab,
    camofox_back,
    _EPOCH_HEADER,
    _adopt_epoch_from_response,
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_snapshot,
    camofox_vision,
)
from tools.browser_tool import _camofox_eval, browser_scroll



def _get_serving_tabs(snapshot_payload, url="https://example.com/page"):
    """Path-aware _get stub: /tabs answers the handback URL guard.

    The guard runs whenever the handback filter is on, so any test that
    exercises a filtered read needs the listing to resolve — otherwise it fails
    closed and the test is measuring the wrong thing.
    """
    def _side_effect(path, params=None, timeout=None, session=None):
        if path == "/tabs":
            tab_id = (session or {}).get("tab_id") or "tab-1"
            session_key = (session or {}).get("session_key") or "s"
            return {"tabs": [{"tabId": tab_id, "listItemId": session_key, "url": url}]}
        return snapshot_payload
    return _side_effect

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
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({
            "snapshot": '- button "Continue" [e9]',
            "refsCount": 1,
        })) as mock_get,
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
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({
            "snapshot": sensitive_snapshot,
            "refsCount": 4,
        })),
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
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({
            "snapshot": private_snapshot,
            "refsCount": 1,
        })),
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
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs(
            {"snapshot": '- heading "Next"', "refsCount": 0},
            url="https://bank.example/login",
        )),
    ):
        result = json.loads(camofox_navigate("https://example.com/next", task_id="agent-task"))

    assert result["success"] is True
    assert managed_session["privacy_filter_after_handback"] is False


def test_console_evaluate_uses_epoch_recovery(managed_session):
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(3)),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({
            "snapshot": '- heading "Signed in"',
            "refsCount": 0,
        })),
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
    managed_session["release_url"] = "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release"
    managed_session["release_headers"] = {"X-Zettlab-Agent-Action-Token": "tok"}
    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    with (
        patch("tools.browser_camofox._drop_session", return_value=managed_session),
        patch("tools.browser_camofox.requests.post", return_value=ok) as mock_post,
        patch("tools.browser_camofox._delete") as mock_delete,
    ):
        result = json.loads(camofox_close("agent-task"))

    assert result == {"success": True, "closed": False, "released": True}
    # The endpoint and credential captured at creation are used verbatim; the
    # teardown path cannot re-read them once the profile scope is gone.
    mock_post.assert_called_once_with(
        "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release",
        json={},
        timeout=5,
        headers={"X-Zettlab-Agent-Action-Token": "tok"},
        allow_redirects=False,
    )
    mock_delete.assert_not_called()


def test_privacy_filter_reduces_snapshot_urls_to_origin(managed_session):
    """URL paths/queries in a post-handback snapshot carry session material.

    The general redaction policy preserves web URL queries, so the handback
    filter itself must reduce every URL in the accessibility tree to its
    origin — OAuth codes, reset tokens and pre-signed links must not reach
    the model.
    """
    managed_session["privacy_filter_after_handback"] = True
    snapshot = (
        '- link "Continue" [e2]:\n'
        "  - /url: https://site.example/callback?code=OAUTHCODE123\n"
        '- link "Download" [e3]:\n'
        "  - /url: https://files.example/doc.pdf?X-Amz-Signature=PRESIGNED456"
    )
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({"snapshot": snapshot, "refsCount": 2})),
    ):
        result = json.loads(camofox_snapshot(task_id="agent-task"))

    assert "OAUTHCODE123" not in result["snapshot"]
    assert "PRESIGNED456" not in result["snapshot"]
    assert "https://site.example/" in result["snapshot"]
    assert "https://files.example/" in result["snapshot"]


def test_privacy_filter_reduces_click_and_back_result_urls(managed_session):
    """Action results must not leak the URL the snapshot filter just hid."""
    managed_session["privacy_filter_after_handback"] = True
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", return_value={"url": "https://site.example/reset?token=RESET789"}),
    ):
        clicked = json.loads(camofox_click("@e4", task_id="agent-task"))
        back = json.loads(camofox_back(task_id="agent-task"))

    assert clicked["url"] == "https://site.example/"
    assert back["url"] == "https://site.example/"


def test_click_result_url_is_untouched_without_privacy_filter(managed_session):
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", return_value={"url": "https://site.example/page?q=fine"}),
    ):
        clicked = json.loads(camofox_click("@e4", task_id="agent-task"))

    assert clicked["url"] == "https://site.example/page?q=fine"


def test_adopting_existing_tab_defaults_privacy_filter_on():
    """A fresh process cannot know whether a human touched the adopted tab.

    A gateway restart right after a handback would otherwise skip the privacy
    filter entirely, so adoption must start filtered; the first agent
    navigation clears it.
    """
    session = {
        "user_id": "hermes_profile",
        "tab_id": None,
        "session_key": "task_opaque",
        "adopt_existing_tab": True,
        "privacy_filter_after_handback": False,
        "epoch": None,
    }
    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab-live", "listItemId": "task_opaque", "epoch": 7}],
        }),
    ):
        adopted = _adopt_existing_tab(session)

    assert adopted["tab_id"] == "tab-live"
    assert adopted["epoch"] == 7
    assert adopted["privacy_filter_after_handback"] is True


def test_url_origin_only_strips_userinfo():
    """Basic-auth credentials in a handback URL must never survive redaction."""
    from tools.browser_camofox import _url_origin_only
    assert _url_origin_only("https://alice:secret@example.com/reset?code=abc") == "https://example.com/"
    assert _url_origin_only("https://user@host.example:8443/path") == "https://host.example:8443/"
    assert _url_origin_only("http://example.com/x?y=1") == "http://example.com/"
    # non-http(s) or unparseable → fully redacted, never echoed back
    assert _url_origin_only("file:///etc/passwd") == "[REDACTED URL]"


def test_internal_proxy_requests_reject_redirects(managed_session):
    """The action token must not follow a cross-origin redirect.

    Requests go out with allow_redirects=False and a 3xx is rejected as an
    error rather than treated as success.
    """
    import requests as _requests
    from tools.browser_camofox import _post

    redirect = MagicMock()
    redirect.status_code = 302
    redirect.headers = {"Location": "https://evil.example/steal"}
    redirect.json.return_value = {}

    captured = {}

    def _capture(url, json=None, timeout=None, headers=None, allow_redirects=None):
        captured["allow_redirects"] = allow_redirects
        captured["token"] = headers.get("X-Zettlab-Agent-Action-Token") if headers else None
        return redirect

    with patch("tools.browser_camofox.requests.post", side_effect=_capture):
        with pytest.raises(Exception):
            _post("/tabs/x/navigate", {"userId": "u"}, session=managed_session)

    assert captured["allow_redirects"] is False


def test_tab_id_from_runtime_must_be_opaque():
    """A malformed tab id must never reach a request path.

    The id is interpolated into /tabs/<id>/... requests that carry the Agent
    action token, so path separators or a query/fragment could re-target those
    privileged requests at other local-server routes.
    """
    from tools.browser_camofox import _tab_path, _validated_tab_id

    assert _validated_tab_id("tab_abc-1.2:3") == "tab_abc-1.2:3"
    for hostile in (
        "../_zettlab/release",
        "tab/../../secret",
        "tab?x=1",
        "tab#frag",
        "tab id",
        "",
        None,
        123,
        "t" * 129,
    ):
        assert _validated_tab_id(hostile) is None, hostile

    assert _tab_path({"tab_id": "tab-1"}, "/snapshot") == "/tabs/tab-1/snapshot"
    with pytest.raises(ValueError):
        _tab_path({"tab_id": "../escape"}, "/snapshot")


def test_ensure_tab_rejects_malformed_runtime_tab_id():
    """A compromised runtime cannot poison the session with a path-bearing id."""
    from tools.browser_camofox import _ensure_tab, _sessions

    _sessions.clear()
    with (
        patch("tools.browser_camofox._get_session", return_value={
            "user_id": "u", "tab_id": None, "session_key": "s", "_lock": None,
        }),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"tabId": "../_zettlab/release"}),
    ):
        with pytest.raises(ValueError):
            _ensure_tab("agent-task")


def test_adopting_existing_tab_rejects_malformed_tab_id():
    """Adoption uses the same validation as creation."""
    session = {
        "user_id": "hermes_profile",
        "tab_id": None,
        "session_key": "task_opaque",
        "adopt_existing_tab": True,
        "privacy_filter_after_handback": False,
        "epoch": None,
    }
    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab/../escape", "listItemId": "task_opaque"}],
        }),
    ):
        adopted = _adopt_existing_tab(session)

    assert adopted["tab_id"] is None


def test_health_check_refuses_to_follow_redirects():
    """The health probe carries the action token and must not follow a 30x."""
    from tools.browser_camofox import check_camofox_available

    captured = {}

    def _capture(url, timeout=None, headers=None, allow_redirects=None):
        captured["allow_redirects"] = allow_redirects
        resp = MagicMock()
        resp.status_code = 302
        return resp

    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox.requests.get", side_effect=_capture),
    ):
        assert check_camofox_available() is False

    assert captured["allow_redirects"] is False


def test_handback_filter_redacts_ipv6_urls():
    """A bracketed IPv6 authority must not slip past the origin reduction."""
    from tools.browser_camofox import _redact_handback_page_state, _url_origin_only

    assert _url_origin_only("https://[2001:db8::1]/cb?code=secret") == "https://[2001:db8::1]/"
    assert _url_origin_only("https://[2001:db8::1]:8443/cb?code=secret") == "https://[2001:db8::1]:8443/"

    filtered = _redact_handback_page_state("link /url: https://[2001:db8::1]/callback?code=abc123")
    assert "abc123" not in filtered
    assert "https://[2001:db8::1]/" in filtered


def test_cleanup_without_session_never_releases_another_turns_lease():
    """A turn that never opened a browser must not release the profile lease.

    cleanup_task_resources() runs at the end of every turn, so an unconditional
    release would tear the runtime down under a concurrent turn — or under a
    human who is mid-takeover.
    """
    from tools.browser_camofox import _sessions, camofox_soft_cleanup

    _sessions.clear()
    released = []
    with (
        patch("tools.browser_camofox._get_camofox_config", return_value={}),
        patch("tools.browser_camofox._local_server_managed", return_value=True),
        patch("tools.browser_camofox._camofox_identity_override", return_value=None),
        patch("tools.browser_camofox.get_camofox_identity", return_value={"user_id": "u", "session_key": "s"}),
        patch("tools.browser_camofox._release_local_server_lease", side_effect=lambda: released.append(1)),
    ):
        assert camofox_soft_cleanup("never-used-browser") is True
    assert released == []


def test_reads_retry_transient_failures_but_mutations_do_not():
    """Bounded backoff for idempotent reads; no blind replay of side effects."""
    from tools.browser_camofox import _get, _post

    unavailable = MagicMock()
    unavailable.status_code = 503
    unavailable.headers = {}
    unavailable.json.return_value = {"error": "starting"}
    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    ok.json.return_value = {"tabs": []}

    get_calls = []

    def _flaky_get(url, params=None, timeout=None, headers=None, allow_redirects=None):
        get_calls.append(url)
        return unavailable if len(get_calls) < 3 else ok

    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
        patch("tools.browser_camofox._auth_headers", return_value={}),
        patch("time.sleep", return_value=None),
        patch("tools.browser_camofox.requests.get", side_effect=_flaky_get),
    ):
        assert _get("/tabs") == {"tabs": []}
    assert len(get_calls) == 3

    post_calls = []

    def _flaky_post(url, json=None, timeout=None, headers=None, allow_redirects=None):
        post_calls.append(url)
        return unavailable

    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
        patch("tools.browser_camofox._auth_headers", return_value={}),
        patch("tools.browser_camofox.requests.post", side_effect=_flaky_post),
    ):
        with pytest.raises(Exception):
            _post("/tabs/t/click", {"ref": "e1"})
    assert len(post_calls) == 1, "a mutation must never be replayed"


def test_vision_discards_screenshot_when_handback_lands_mid_call():
    """The epoch that enables the filter arrives with the screenshot response."""
    from tools.browser_camofox import camofox_vision

    session = {
        "user_id": "u",
        "tab_id": "tab-1",
        "session_key": "s",
        "privacy_filter_after_handback": False,
        "epoch": 1,
        "_lock": None,
    }

    def _raw(path, params=None, timeout=None, session=None):
        # The response carries the post-handback epoch.
        session["privacy_filter_after_handback"] = True
        resp = MagicMock()
        resp.content = b"\x89PNG-private-screen"
        return resp

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._get_raw", side_effect=_raw),
    ):
        result = json.loads(camofox_vision("what is on screen?", task_id="agent-task"))

    assert result["success"] is False
    assert "blocked after human control" in result["error"]


def test_navigate_auto_snapshot_is_filtered_after_mid_call_handback():
    """navigate's bonus snapshot goes through the same filter as camofox_snapshot.

    A human taking over and handing back between the navigate response and the
    snapshot response would otherwise put their typed credentials straight into
    the tool result.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u",
        "tab_id": "tab-1",
        "session_key": "s",
        "privacy_filter_after_handback": False,
        "epoch": 1,
        "_lock": None,
    }

    def _get_with_handback(path, params=None, timeout=None, session=None):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "https://example.com/page"}]}
        # The snapshot response carries the post-handback epoch.
        session["privacy_filter_after_handback"] = True
        return {"snapshot": "textbox \"Password\" value=\"hunter2\"", "refsCount": 1}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/", "title": "Example"}),
        patch("tools.browser_camofox._get", side_effect=_get_with_handback),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert "hunter2" not in json.dumps(result)
    assert "REDACTED" in result["snapshot"]


def test_cleanup_never_reaches_into_another_profiles_session():
    """A task id is not owner-scoped, so it can never authorize a drop.

    The API's session_id becomes the effective task id, so two profiles can
    carry the same one. Cleanup runs after the profile secret scope is gone,
    where the identity cannot be recomputed — dropping on a task-id match alone
    would delete another user's live browser state and release their lease.
    """
    from agent.secret_scope import UnscopedSecretError
    from tools.browser_camofox import _drop_session, _sessions, has_camofox_session

    _sessions.clear()
    _sessions["profileA\x00sessA\x00shared-task"] = {
        "user_id": "profileA", "session_key": "sessA", "tab_id": "tab-a",
        "managed": True, "local_server_managed": True, "task_id": "shared-task",
        "last_used_at": time.monotonic(),
    }

    def _fail_closed(*args, **kwargs):
        raise UnscopedSecretError("no secret scope installed")

    with (
        patch("tools.browser_camofox._camofox_identity_override", side_effect=_fail_closed),
        patch("tools.browser_camofox.get_camofox_identity", side_effect=_fail_closed),
    ):
        assert has_camofox_session("shared-task") is False
        assert _drop_session("shared-task") is None
    assert len(_sessions) == 1, "a scope-less cleanup must not drop anyone's session"
    _sessions.clear()


def test_idle_sessions_are_reclaimed_with_their_captured_release_context():
    """The scope-less path is bounded by a timer, not by guessing.

    Nothing else can reclaim a session whose profile scope is gone, so without
    this both the entry and the local-server lease behind it would live for the
    life of the process — unbounded resident state on a 2 GB device. The
    release endpoint and credential come from the session because they cannot
    be re-read at this point.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _drop_session, _sessions

    _sessions.clear()
    _sessions["profileA\x00sessA\x00old-task"] = {
        "user_id": "profileA", "session_key": "sessA", "tab_id": "tab-a",
        "managed": True, "local_server_managed": True, "task_id": "old-task",
        "release_url": "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release",
        "release_headers": {"X-Zettlab-Agent-Action-Token": "tok-a"},
        "last_used_at": time.monotonic() - (mod._SESSION_IDLE_TTL_SECONDS + 1),
    }

    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    with (
        patch("tools.browser_camofox._camofox_identity_override", return_value=None),
        patch("tools.browser_camofox.get_camofox_identity", return_value={"user_id": "x", "session_key": "y"}),
        patch("tools.browser_camofox._get_camofox_config", return_value={}),
        patch("tools.browser_camofox.requests.post", return_value=ok) as mock_post,
    ):
        _drop_session("unrelated-task")

    assert _sessions == {}
    assert mock_post.call_args.kwargs["headers"] == {"X-Zettlab-Agent-Action-Token": "tok-a"}
    assert mock_post.call_args.args[0].endswith("/_zettlab/release")
    _sessions.clear()


def test_navigate_keeps_privacy_filter_when_handback_lands_mid_navigation():
    """A handback during the navigate call must not be cleared by that call.

    The response carries the new epoch, so the server will not flag the next
    read as stale either — clearing the filter here would put the values the
    human just typed in front of the model.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _post_with_handback(path, body=None, timeout=None, session=None):
        session["epoch"] = 4  # the human handed back while this was in flight
        session["privacy_filter_after_handback"] = True
        return {"url": "https://example.com/", "title": "Example"}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", side_effect=_post_with_handback),
        patch("tools.browser_camofox._get", return_value={"snapshot": "textbox \"OTP\" value=\"123456\"", "refsCount": 1}),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert session["privacy_filter_after_handback"] is True
    assert "123456" not in json.dumps(result)


def test_navigate_clears_privacy_filter_on_a_quiet_navigation():
    """The control case: no epoch change means the Agent moved the page itself."""
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 3, "_lock": None,
    }
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/", "title": "Example"}),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs(
            {"snapshot": "- heading \"Example\"", "refsCount": 1},
            url="https://bank.example/login",
        )),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert session["privacy_filter_after_handback"] is False
    assert "Example" in result["snapshot"]


def test_multi_turn_readoption_does_not_look_like_a_handback():
    """Per-turn soft cleanup must not degrade ordinary multi-turn browsing.

    cleanup_task_resources drops the in-process session at the end of every
    turn, so treating every re-adoption as a possible handback would blank out
    every form control and block vision/eval from the second turn on.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _adopt_existing_tab, _remember_tab_epoch

    mod._remembered_tab_epochs.clear()
    live = {
        "user_id": "hermes_profile", "session_key": "task_opaque",
        "tab_id": "tab-live", "epoch": 7,
    }
    _remember_tab_epoch(live)


    # Same epoch as the last turn: nobody took over, so browsing continues.
    session_same = {
        "user_id": "hermes_profile", "session_key": "task_opaque", "tab_id": None,
        "adopt_existing_tab": True, "privacy_filter_after_handback": False, "epoch": None,
    }
    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab-live", "listItemId": "task_opaque", "epoch": 7}],
        }),
    ):
        adopted = _adopt_existing_tab(session_same)
    assert adopted["privacy_filter_after_handback"] is False

    # Epoch advanced while this process was not looking: a handback did happen.
    session_moved = {
        "user_id": "hermes_profile", "session_key": "task_opaque", "tab_id": None,
        "adopt_existing_tab": True, "privacy_filter_after_handback": False, "epoch": None,
    }
    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab-live", "listItemId": "task_opaque", "epoch": 8}],
        }),
    ):
        adopted = _adopt_existing_tab(session_moved)
    assert adopted["privacy_filter_after_handback"] is True
    mod._remembered_tab_epochs.clear()


def test_navigate_result_url_is_filtered_after_mid_call_handback():
    """The page the human landed on is reported by navigate itself.

    Filtering only the bonus snapshot would still hand the model an OAuth
    callback or reset link in full.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _post_with_handback(path, body=None, timeout=None, session=None):
        session["epoch"] = 4
        session["privacy_filter_after_handback"] = True
        return {"url": "https://idp.example/callback?code=SECRETCODE", "title": "Signed in as alice@example.com"}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", side_effect=_post_with_handback),
        patch("tools.browser_camofox._get", return_value={"snapshot": "- heading \"Welcome\"", "refsCount": 1}),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    body = json.dumps(result)
    assert "SECRETCODE" not in body
    assert "alice@example.com" not in body
    assert result["url"] == "https://idp.example/"


def test_lease_release_is_retried_and_queued_until_it_lands():
    """The session is already gone, so this context is the only way to free it.

    A transient failure that only logged would strand a long-lived
    local-server lease on a 2 GB device with nothing left to retry it.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _release_local_server_lease, _flush_pending_lease_releases

    mod._pending_lease_releases.clear()
    session = {
        "release_url": "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release",
        "release_headers": {"X-Zettlab-Agent-Action-Token": "tok"},
    }

    unavailable = MagicMock()
    unavailable.status_code = 503
    unavailable.headers = {}
    unavailable.json.return_value = {}

    # The reclaim thread is stubbed out so the queue can be inspected; its own
    # handoff is covered by test_queued_release_always_has_a_consumer.
    with (
        patch("time.sleep", return_value=None),
        patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None),
        patch("tools.browser_camofox.requests.post", return_value=unavailable) as failing,
    ):
        _release_local_server_lease(session)
    assert failing.call_count > 1, "a transient failure must be retried, not just logged"
    assert len(mod._pending_lease_releases) == 1, "an undelivered release must stay queued"

    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    with patch("tools.browser_camofox.requests.post", return_value=ok) as succeeding:
        _flush_pending_lease_releases()
    succeeding.assert_called_once()
    assert mod._pending_lease_releases == []
    mod._pending_lease_releases.clear()


def test_tab_id_rejects_dot_segments():
    """`.` and `..` are allowed characters but are path segments, not ids.

    quote() leaves dots alone, so `/tabs/../snapshot` would survive to whatever
    normalizes the path next — carrying the Agent action token out of the tab
    API.
    """
    from tools.browser_camofox import _validated_tab_id

    for hostile in ("..", ".", "...", " .. "):
        assert _validated_tab_id(hostile) is None, hostile
    # A dot inside a real id stays legal.
    assert _validated_tab_id("tab.1") == "tab.1"


def test_readoption_restores_the_filter_a_mid_handback_turn_left_on():
    """Remembering the epoch alone would clear a filter that must stay on.

    A handback detected during a turn leaves the filter on at an epoch the
    server also reports, so "epoch unchanged" does not mean "nothing happened".
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _adopt_existing_tab, _remember_tab_epoch

    mod._remembered_tab_epochs.clear()
    # The turn ended while the page was still under the handback filter.
    _remember_tab_epoch({
        "user_id": "hermes_profile", "session_key": "task_opaque",
        "tab_id": "tab-live", "epoch": 9, "privacy_filter_after_handback": True,
    })

    session = {
        "user_id": "hermes_profile", "session_key": "task_opaque", "tab_id": None,
        "adopt_existing_tab": True, "privacy_filter_after_handback": False, "epoch": None,
    }
    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:8080"),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab-live", "listItemId": "task_opaque", "epoch": 9}],
        }),
    ):
        adopted = _adopt_existing_tab(session)

    assert adopted["privacy_filter_after_handback"] is True
    mod._remembered_tab_epochs.clear()


def test_recovery_refuses_to_snapshot_a_blocked_page():
    """The human may hand back on cloud metadata or an intranet page.

    The deleted resume handshake validated this before acking; the recovery
    snapshot must apply the same guard browser_navigate applies to the Agent's
    own navigations.
    """
    from tools.browser_camofox import _retryable_control_result

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }
    stale = _epoch_stale(epoch=4)

    with (
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", return_value={
            "tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://169.254.169.254/latest/meta-data/"}],
        }) as mock_get,
        patch("tools.browser_camofox._redact_handback_page_state") as redact,
    ):
        result = json.loads(_retryable_control_result(stale, session=session))

    assert result.get("blocked_page") is True
    assert "snapshot" not in result
    redact.assert_not_called()
    # Only the /tabs lookup ran; the snapshot was never requested.
    assert mock_get.call_count == 1

    # The guard must discriminate, not just fail closed — otherwise a broken
    # import would make the assertions above pass while blocking everything.
    from tools.browser_camofox import _recovery_target_allowed

    assert _recovery_target_allowed("https://example.com/page") is True
    assert _recovery_target_allowed("http://169.254.169.254/latest/meta-data/") is False
    assert _recovery_target_allowed("http://metadata.google.internal/") is False
    assert _recovery_target_allowed("http://192.168.1.10/admin") is False


def test_snapshot_refuses_a_blocked_page_after_handback():
    """The success path needs the same guard the 409 recovery path has.

    If the Agent's first call after a handback is browser_snapshot, the proxy
    answers 200 and only advances the epoch through the header — so a human who
    left the tab on cloud metadata would have its contents redacted but still
    delivered, along with actionable refs.
    """
    from tools.browser_camofox import camofox_snapshot

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _get_with_handback(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://169.254.169.254/latest/meta-data/"}]}
        session["privacy_filter_after_handback"] = True  # the epoch arrived with this response
        return {"snapshot": "- text \"iam credentials\"", "refsCount": 3}

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", side_effect=_get_with_handback),
    ):
        result = json.loads(camofox_snapshot(task_id="agent-task"))

    assert result["success"] is False
    assert "not allowed to read" in result["error"]
    assert "iam credentials" not in json.dumps(result)


def test_fragment_navigation_does_not_clear_the_handback_filter():
    """A same-document navigation keeps the DOM the human typed into.

    It also does not advance the handback epoch, so "the epoch stood still" is
    not evidence the Agent left the page.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 4, "_lock": None,
    }
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://bank.example/transfer#step2", "title": "Transfer"}),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs(
            {"snapshot": "textbox \"Account\" value=\"12345678\"", "refsCount": 1},
            url="https://bank.example/transfer",
        )),
    ):
        result = json.loads(camofox_navigate("https://bank.example/transfer#step2", task_id="agent-task"))

    assert session["privacy_filter_after_handback"] is True
    assert "12345678" not in json.dumps(result)


def test_managed_tab_read_without_epoch_header_fails_closed():
    """A local-server that stops sending the epoch must not silently disarm it.

    The epoch is the only signal that a human touched the page, so a successful
    managed tab response without one is a protocol failure, and the safe
    reading is "assume the page changed".
    """
    from tools.browser_camofox import _adopt_epoch_from_response

    session = {"epoch": 5, "local_server_managed": True, "privacy_filter_after_handback": False}
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 200

    _adopt_epoch_from_response(session, resp, tab_operation=True)
    assert session["privacy_filter_after_handback"] is True

    # An error envelope carries no page data, so it says nothing either way.
    session["privacy_filter_after_handback"] = False
    resp.status_code = 503
    _adopt_epoch_from_response(session, resp, tab_operation=True)
    assert session["privacy_filter_after_handback"] is False

    # Neither does a non-tab path.
    resp.status_code = 200
    _adopt_epoch_from_response(session, resp, tab_operation=False)
    assert session["privacy_filter_after_handback"] is False


def test_queued_release_always_has_a_consumer():
    """Enqueue and the worker's exit decision share one lock.

    Without that handoff a release queued just as the worker was leaving would
    sit there forever: the old thread still reports is_alive(), so no
    replacement starts, and then it exits.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _queue_pending_lease_release

    mod._pending_lease_releases.clear()
    mod._maintenance_worker = None
    try:
        _queue_pending_lease_release("http://127.0.0.1:9377/x/_zettlab/release", {})
        assert mod._maintenance_worker is not None, "a queued release must have a consumer"
        assert mod._maintenance_worker.daemon is True
    finally:
        mod._pending_lease_releases.clear()


def test_handback_filter_redacts_ipv6_with_userinfo():
    """Basic-auth credentials in front of a bracketed IPv6 host.

    Handling userinfo and IPv6 separately still truncated the match at the "@",
    leaving the path and query — the OAuth code — in the text verbatim.
    """
    from tools.browser_camofox import _reduce_urls_to_origin

    filtered = _reduce_urls_to_origin("go https://alice:secret@[2001:db8::1]/callback?code=abc123 now")
    assert "abc123" not in filtered
    assert "secret" not in filtered
    assert "https://[2001:db8::1]/" in filtered


def test_navigate_snapshot_is_withheld_for_a_blocked_page():
    """navigate's inline snapshot needs the same guard as camofox_snapshot.

    Redaction only strips form values; ordinary page text from cloud metadata
    or an intranet host would still reach the model.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _get_with_handback(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://169.254.169.254/latest/meta-data/"}]}
        session["privacy_filter_after_handback"] = True
        return {"snapshot": "- text \"iam credentials\"", "refsCount": 2}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/", "title": "Example"}),
        patch("tools.browser_camofox._get", side_effect=_get_with_handback),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert result.get("snapshot_withheld") is True
    assert "iam credentials" not in json.dumps(result)
