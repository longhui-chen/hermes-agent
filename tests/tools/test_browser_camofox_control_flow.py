"""Flow tests for the epoch-based takeover/handback synchronization.

The local-server proxy keeps one monotonic epoch per tab that increments when
human control ends. Hermes declares its last-seen epoch on tab operations; a
mismatch returns 409 browser_epoch_stale and recovery is a single stateless
step: adopt the new epoch, take one privacy-filtered snapshot, retry.
"""

import contextlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.browser_camofox import (
    _adopt_existing_tab,
    _run_pending_teardowns,
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

@pytest.fixture(autouse=True)
def _clear_response_facts():
    """Reset the per-response thread-locals around every test.

    Production resets them at the start of each transport call, but tests mock
    that layer away — without this a test can pass on a neighbour's leftover.
    """
    import tools.browser_camofox as mod

    for attr in ("started_handback", "epoch_verified"):
        if hasattr(mod._response_facts, attr):
            delattr(mod._response_facts, attr)
    yield
    for attr in ("started_handback", "epoch_verified"):
        if hasattr(mod._response_facts, attr):
            delattr(mod._response_facts, attr)


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


def _http_error(status: int, payload: dict) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    return requests.HTTPError(response=response)


def _epoch_stale(epoch: int = 3) -> requests.HTTPError:
    """A refusal from local-server.

    ``epoch`` is what an older proxy used to include and what a hostile one
    could still send; nothing on this side may act on it, so the tests keep
    sending it and assert it is ignored.
    """
    return _http_error(
        409,
        {
            "error": "browser_epoch_stale",
            "message": "page state changed after human control; snapshot the tab before continuing",
            "epoch": epoch,
        },
    )


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
    # The privacy filter is on, and the epoch did NOT come from the refusal:
    # this fixture patches _get, so no response header reaches the adoption
    # path and the session keeps its own value. In production the recovery
    # snapshot's header is what moves it.
    assert managed_session["epoch"] == 2
    assert managed_session["privacy_filter_after_handback"] is True
    # The blocked-page guard queries /tabs before and after the snapshot, so
    # assert the snapshot happened rather than that it was last.
    assert any(call.args[:1] == ("/tabs/tab-agent/snapshot",) for call in mock_get.call_args_list)
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
    # The refusal's own epoch is never adopted: taking it would clear the
    # barrier without the snapshot that carries the human's page state. The
    # snapshot failed here, so this session stays on its old epoch and the next
    # attempt is refused again.
    assert managed_session["epoch"] == 2


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
    response.status_code = 200
    response.headers = {_EPOCH_HEADER: "3"}
    _adopt_epoch_from_response(managed_session, response)

    assert managed_session["epoch"] == 3
    assert managed_session["privacy_filter_after_handback"] is True


def test_epoch_header_same_value_keeps_filter_off(managed_session):
    response = MagicMock()
    response.status_code = 200
    response.headers = {_EPOCH_HEADER: "2"}
    _adopt_epoch_from_response(managed_session, response)

    assert managed_session["epoch"] == 2
    assert managed_session["privacy_filter_after_handback"] is False


def test_first_epoch_sighting_does_not_enable_filter(managed_session):
    managed_session["epoch"] = None
    response = MagicMock()
    response.status_code = 200
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
        # The release is scheduled behind a quiet window; run it now.
        _run_pending_teardowns(force=True)

    assert result == {"success": True, "closed": False, "released": True}
    # The endpoint and credential captured at creation are used verbatim; the
    # teardown path cannot re-read them once the profile scope is gone.
    mock_post.assert_called_once_with(
        "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release",
        json={},
        timeout=5,
        headers={"X-Zettlab-Agent-Action-Token": "tok"},
        allow_redirects=False,
        proxies={"http": None, "https": None, "all": None},
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

    def _capture(url, json=None, timeout=None, headers=None, allow_redirects=None, proxies=None):
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

    def _capture(url, timeout=None, headers=None, allow_redirects=None, proxies=None):
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

    def _flaky_get(url, params=None, timeout=None, headers=None, allow_redirects=None, **_kw):
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

    def _flaky_post(url, json=None, timeout=None, headers=None, allow_redirects=None, **_kw):
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
        _run_pending_teardowns(force=True)

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


def test_lease_release_is_scheduled_and_retried_until_it_lands():
    """The release is deferred and retried; it is never fire-and-forget.

    The session is dropped before this runs, so the queued entry is the only
    remaining handle on a long-lived local-server lease — a transient failure
    that merely logged would strand it on a 2 GB device.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _release_local_server_lease

    mod._pending_lease_releases.clear()
    session = {
        "release_url": "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release",
        "release_headers": {"X-Zettlab-Agent-Action-Token": "tok"},
        "release_owner": "profileA\x00digest",
    }

    unavailable = MagicMock()
    unavailable.status_code = 503
    unavailable.headers = {}
    unavailable.json.return_value = {}

    with patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None):
        _release_local_server_lease(session)
        # Deferred: nothing has been sent yet, and the entry is waiting behind
        # its quiet window.
        assert len(mod._pending_lease_releases) == 1
        assert mod._pending_lease_releases[0]["ready_at"] > time.monotonic()

        with patch("tools.browser_camofox.requests.post", return_value=unavailable) as failing:
            _run_pending_teardowns(force=True)
        assert failing.call_count == 1, "one attempt per run, not a retry loop"
        assert len(mod._pending_lease_releases) == 1, "an undelivered release stays queued"

        ok = MagicMock()
        ok.status_code = 200
        ok.headers = {}
        with patch("tools.browser_camofox.requests.post", return_value=ok) as succeeding:
            _run_pending_teardowns(force=True)
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


def test_soft_cleanup_never_releases_a_shared_profile_lease():
    """Turn end says nothing about whether the browser is still in use.

    A vision call runs for minutes and a human takeover for longer, so any
    timer started here is a guess. The entry stays and its last-use stamp is
    what the idle sweep acts on.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_soft_cleanup

    owner = "profileA\x00digest"
    identity = {"user_id": "profileA", "session_key": "sess"}
    mod._sessions.clear()
    mod._pending_lease_releases.clear()
    # The cache key carries the credential-derived owner: two profiles handed
    # the same explicit identity must not share an entry.
    key = mod._session_cache_key("turn-a", identity, owner)
    mod._sessions[key] = {
        "user_id": "profileA", "session_key": "sess", "tab_id": "tab-a",
        "managed": True, "local_server_managed": True, "task_id": "turn-a",
        "release_url": "http://127.0.0.1:9377/x/_zettlab/release",
        "release_headers": {}, "release_owner": owner,
        "last_used_at": time.monotonic(),
    }

    with (
        patch("tools.browser_camofox._get_camofox_config", return_value={}),
        patch("tools.browser_camofox._camofox_identity_override", return_value=None),
        patch("tools.browser_camofox.get_camofox_identity", return_value=identity),
        patch("tools.browser_camofox._release_owner_key", return_value=owner),
        patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None),
        patch("tools.browser_camofox.requests.post") as mock_post,
    ):
        assert camofox_soft_cleanup("turn-a") is True
        mock_post.assert_not_called()
        assert mod._pending_lease_releases == [], "turn end must not schedule a release"
        assert key in mod._sessions, "the entry stays for the idle sweep to judge"
    mod._sessions.clear()


def test_idle_direct_session_is_really_closed():
    """A direct Camofox session owns a server-side session, not a lease.

    Reclaiming it through the local-server release path would post to an
    endpoint that does not exist there and leave the tab and browser running.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _teardown_session

    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    with patch("tools.browser_camofox.requests.delete", return_value=ok) as mock_delete:
        _teardown_session({
            "user_id": "hermes_abc123", "managed": False, "local_server_managed": False,
            # Captured at creation: the maintenance thread has no profile scope
            # to re-read these from.
            "delete_base": "http://127.0.0.1:9377",
            "delete_headers": {"Authorization": "Bearer direct-key"},
        })
        _run_pending_teardowns(force=True)
    mock_delete.assert_called_once_with(
        "http://127.0.0.1:9377/sessions/hermes_abc123",
        timeout=5,
        headers={"Authorization": "Bearer direct-key"},
        allow_redirects=False,
        proxies={"http": None, "https": None, "all": None},
    )

    # A managed-persistence profile must survive instead.
    with patch("tools.browser_camofox.requests.delete") as mock_delete:
        _teardown_session({"user_id": "profileA", "managed": True, "local_server_managed": False})
        _run_pending_teardowns(force=True)
    mock_delete.assert_not_called()


def test_handback_filter_redacts_scoped_ipv6_urls():
    """RFC 6874 zone identifiers contain letters outside the hex alphabet."""
    from tools.browser_camofox import _reduce_urls_to_origin

    filtered = _reduce_urls_to_origin("go https://u:p@[fe80::1%25eth0]/cb?code=abc123 now")
    assert "abc123" not in filtered
    assert "u:p@" not in filtered
    assert "https://[fe80::1%25eth0]/" in filtered


def test_get_images_refuses_a_blocked_page_after_handback():
    """alt/src are page content the redaction pass does not remove."""
    from tools.browser_camofox import camofox_get_images

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _get_with_handback(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://169.254.169.254/latest/meta-data/"}]}
        session["privacy_filter_after_handback"] = True
        return {"snapshot": '- image "internal-topology":\n  - /url: http://10.0.0.5/diagram.png', "refsCount": 1}

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", side_effect=_get_with_handback),
    ):
        result = json.loads(camofox_get_images(task_id="agent-task"))

    assert result["success"] is False
    assert "internal-topology" not in json.dumps(result)


def test_lease_owner_is_the_profile_not_the_url():
    """In multiplex every profile shares one loopback CAMOFOX_URL.

    Grouping holders by URL would make profile A see profile B as its own
    holder, skip the release and lose A's only release context.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _profile_still_in_use, _release_owner_key

    a = _release_owner_key("profileA", {"X-Zettlab-Agent-Action-Token": "token-a"})
    b = _release_owner_key("profileB", {"X-Zettlab-Agent-Action-Token": "token-b"})
    assert a != b
    # Same profile, same credential → same owner; the raw token is not stored.
    assert a == _release_owner_key("profileA", {"X-Zettlab-Agent-Action-Token": "token-a"})
    assert "token-a" not in a

    mod._sessions.clear()
    mod._sessions["profileB\x00sess\x00turn-b"] = {
        "user_id": "profileB", "session_key": "sess", "task_id": "turn-b",
        "managed": True, "local_server_managed": True, "release_owner": b,
        "last_used_at": time.monotonic(),
    }
    assert _profile_still_in_use(a) is False, "another profile is not a holder"
    assert _profile_still_in_use(b) is True
    mod._sessions.clear()


def test_release_is_serialized_against_session_registration():
    """The holder check and the release must be one critical section.

    Otherwise another turn registers and starts the runtime in between, and the
    release tears it down under them — or under a human mid-takeover.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _owner_lock, _teardown_session

    owner = "profileA\x00digest"
    mod._sessions.clear()
    released = []

    def _record(session=None):
        # The lock is reentrant for this thread, so exclusivity is checked from
        # another one: it must not be able to take it while the release runs.
        import threading as _threading

        taken = []

        def _try():
            got = _owner_lock(owner).acquire(blocking=False)
            taken.append(got)
            if got:
                _owner_lock(owner).release()

        probe = _threading.Thread(target=_try)
        probe.start()
        probe.join()
        assert taken == [False], "release ran outside the owner lock"
        released.append(1)

    with patch("tools.browser_camofox._release_local_server_lease", side_effect=_record):
        _teardown_session({
            "user_id": "profileA", "managed": True, "local_server_managed": True,
            "release_owner": owner, "release_url": "http://127.0.0.1:9377/x/_zettlab/release",
        })
    assert released == [1]
    mod._sessions.clear()


def test_queued_release_is_dropped_when_the_profile_is_in_use_again():
    """A turn that started while the release was queued now owns the runtime.

    Completing the queued release would tear it down under that turn — the
    retry path has to re-check holders, not just re-send.
    """
    import tools.browser_camofox as mod

    owner = "profileA\x00digest"
    mod._sessions.clear()
    mod._pending_lease_releases[:] = [{
        "kind": "release", "url": "http://127.0.0.1:9377/x/_zettlab/release",
        "headers": {}, "owner": owner, "attempts": 0, "ready_at": 0.0,
    }]
    mod._sessions["profileA\x00sess\x00new-turn"] = {
        "user_id": "profileA", "session_key": "sess", "task_id": "new-turn",
        "managed": True, "local_server_managed": True, "release_owner": owner,
        "last_used_at": time.monotonic(),
    }

    with patch("tools.browser_camofox.requests.post") as mock_post:
        _run_pending_teardowns(force=True)
    mock_post.assert_not_called()
    assert mod._pending_lease_releases == []
    mod._sessions.clear()




def test_owner_lock_is_never_replaced_while_held():
    """Evicting a held lock would split the critical section in two."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import _owner_lock

    mod._owner_locks.clear()
    mod._sessions.clear()
    owner = "profileA\x00digest"
    held = _owner_lock(owner, reserve=True)
    held.acquire()
    try:
        # Push well past the cap with unrelated owners.
        for i in range(mod._MAX_OWNER_LOCKS + 5):
            _owner_lock(f"filler-{i}")
        assert _owner_lock(owner) is held, "a held lock must never be replaced"
    finally:
        held.release()
        mod._owner_lock_refs.clear()
        mod._owner_locks.clear()


def test_managed_tab_creation_without_epoch_fails_closed():
    """POST /tabs establishes the baseline epoch for the new tab.

    Without it a later response's epoch looks like the first one ever seen and
    is taken as a safe baseline, so a takeover between creation and the first
    read would go unnoticed.
    """
    from tools.browser_camofox import _adopt_epoch_from_response

    session = {"epoch": None, "local_server_managed": True, "privacy_filter_after_handback": False}
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 200
    _adopt_epoch_from_response(session, resp, tab_operation=True)
    assert session["privacy_filter_after_handback"] is True


def test_blocked_capture_url_wins_over_a_later_lookup():
    """The URL the capture reports describes where the content came from.

    A separate /tabs lookup can only say where the tab is now, so a page that
    bounced back to a public address between capture and lookup would otherwise
    let the restricted content through.
    """
    from tools.browser_camofox import camofox_snapshot

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 3, "_lock": None,
    }

    def _get_with_handback(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            # Already back on a public page by the time this is asked.
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "https://example.com/"}]}
        session["privacy_filter_after_handback"] = True
        return {
            "snapshot": "- text \"iam credentials\"",
            "refsCount": 1,
            "url": "http://169.254.169.254/latest/meta-data/",
        }

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", side_effect=_get_with_handback),
    ):
        result = json.loads(camofox_snapshot(task_id="agent-task"))

    assert result["success"] is False
    assert "iam credentials" not in json.dumps(result)


def test_teardown_retries_a_server_error():
    """A 5xx is the server failing to answer, not an answer.

    Treating an internal error as definitive drops a teardown whose runtime is
    still alive, with nothing left to close it.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _queue_pending_teardown

    mod._pending_lease_releases.clear()
    failing = MagicMock()
    failing.status_code = 500
    failing.headers = {}
    failing.json.return_value = {}

    with patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None):
        _queue_pending_teardown("release", "http://127.0.0.1:9377/x/_zettlab/release", {})
        with patch("tools.browser_camofox.requests.post", return_value=failing):
            _run_pending_teardowns(force=True)
        assert len(mod._pending_lease_releases) == 1, "a 5xx must stay queued"
    mod._pending_lease_releases.clear()


def test_teardown_queue_merges_repeats_instead_of_dropping_work():
    """Repeat teardowns of one target are the same work.

    Dropping the oldest entry to stay under a cap would throw away the only
    handle that can close a runtime.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _queue_pending_teardown

    mod._pending_lease_releases.clear()
    with patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None):
        for _ in range(mod._MAX_PENDING_LEASE_RELEASES * 2):
            _queue_pending_teardown("release", "http://127.0.0.1:9377/same/_zettlab/release", {})
    assert len(mod._pending_lease_releases) == 1
    mod._pending_lease_releases.clear()


def test_owner_lock_survives_eviction_pressure_while_reserved():
    """Between lookup and acquire the lock must not become evictable."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import _held_owner_lock, _owner_lock

    mod._owner_locks.clear()
    mod._owner_lock_refs.clear()
    mod._sessions.clear()
    owner = "profileA\x00digest"
    with _held_owner_lock(owner) as held:
        for i in range(mod._MAX_OWNER_LOCKS + 5):
            _owner_lock(f"filler-{i}")
        assert _owner_lock(owner) is held
    assert mod._owner_lock_refs == {}
    mod._owner_locks.clear()


def test_release_teardowns_are_not_merged_across_profiles():
    """In multiplex every profile posts to the same /_zettlab/release.

    Merging on the URL alone would overwrite another profile's credential —
    the only thing that can release its lease.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _queue_pending_teardown

    url = "http://127.0.0.1:9377/internal/browser/camofox/_zettlab/release"
    mod._pending_lease_releases.clear()
    with patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None):
        _queue_pending_teardown("release", url, {"X-Zettlab-Agent-Action-Token": "tok-a"}, owner="profileA\x00a")
        _queue_pending_teardown("release", url, {"X-Zettlab-Agent-Action-Token": "tok-b"}, owner="profileB\x00b")

    assert len(mod._pending_lease_releases) == 2
    tokens = {e["headers"]["X-Zettlab-Agent-Action-Token"] for e in mod._pending_lease_releases}
    assert tokens == {"tok-a", "tok-b"}, "each profile keeps its own release credential"
    mod._pending_lease_releases.clear()


def test_concurrent_turns_on_one_identity_create_one_tab():
    """Parent and subagent turns share a browser identity but not a task id.

    Without a shared critical section both would create a tab under the same
    listItemId, and adoption could then only guess which one is "the" tab.
    """
    import threading

    import tools.browser_camofox as mod
    from tools.browser_camofox import _ensure_tab

    mod._owner_locks.clear()
    mod._owner_lock_refs.clear()
    created = []
    listed = []

    def _post(path, body=None, timeout=None, session=None):
        created.append(body["listItemId"])
        listed.append({"tabId": f"tab-{len(created)}", "listItemId": body["listItemId"]})
        return {"tabId": f"tab-{len(created)}"}

    def _get(path, params=None, timeout=None, session=None, **kwargs):
        return {"tabs": list(listed)}

    def _session_for(task_id):
        return {
            "user_id": "profileA", "session_key": "shared-session", "tab_id": None,
            "adopt_existing_tab": True, "privacy_filter_after_handback": False,
            "epoch": None, "task_id": task_id, "_lock": threading.Lock(),
        }

    sessions = {"parent": _session_for("parent"), "sub": _session_for("sub")}
    barrier = threading.Barrier(2)

    def _run(task_id):
        barrier.wait()
        with (
            patch("tools.browser_camofox._get_session", return_value=sessions[task_id]),
            patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
            patch("tools.browser_camofox._post", side_effect=_post),
            patch("tools.browser_camofox._get", side_effect=_get),
        ):
            _ensure_tab(task_id)

    threads = [threading.Thread(target=_run, args=(t,)) for t in ("parent", "sub")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1, f"one tab per browser identity, got {created}"
    assert sessions["parent"]["tab_id"] == sessions["sub"]["tab_id"]
    mod._owner_locks.clear()


def test_concurrent_navigate_cannot_unfilter_an_in_flight_capture():
    """Two turns share one session dict, so the flag can move mid-read.

    A capture taken while the filter was on must stay filtered even if a
    navigate clears it before the response is processed.
    """
    from tools.browser_camofox import camofox_snapshot

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 4, "_lock": None,
    }

    def _get_racing_navigate(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "https://example.com/"}]}
        # The other turn's navigate lands here, clearing the shared flag.
        session["privacy_filter_after_handback"] = False
        return {"snapshot": "textbox \"Password\" value=\"hunter2\"", "refsCount": 1}

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", side_effect=_get_racing_navigate),
    ):
        result = json.loads(camofox_snapshot(task_id="agent-task"))

    assert "hunter2" not in json.dumps(result)


def test_navigate_keeps_the_filter_when_the_epoch_was_not_verified():
    """A protocol downgrade must not read as "nothing happened".

    Without a verified epoch on this very response, "the epoch did not move"
    only means nothing was reported — not that nobody took over.
    """
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 4, "_lock": None,
        "local_server_managed": True, "last_epoch_verified": False,
    }
    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://other.example/", "title": "Other"}),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs(
            {"snapshot": "- heading \"Other\"", "refsCount": 1},
            url="https://bank.example/login",
        )),
    ):
        json.loads(camofox_navigate("https://other.example/", task_id="agent-task"))

    assert session["privacy_filter_after_handback"] is True


def test_handback_filter_redacts_urls_with_parenthesised_paths():
    """Parentheses are legal in a path, and the tail after them is the payload."""
    from tools.browser_camofox import _reduce_urls_to_origin

    filtered = _reduce_urls_to_origin("go https://example.com/(S(secret))/callback?code=abc123 now")
    assert "abc123" not in filtered
    assert "secret" not in filtered
    assert "https://example.com/" in filtered


def test_epoch_verification_is_response_local_not_session_state():
    """A concurrent response must not decide this navigate's outcome.

    Two turns share the session dict, so a click carrying a valid epoch could
    otherwise flip a shared "verified" flag back on between this navigate's
    unverified response and the check that clears the filter.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 4, "_lock": None,
        "local_server_managed": True,
    }
    # Whatever another turn last did on its own thread must not leak in.
    mod._response_facts.epoch_verified = True

    def _post_without_epoch(path, body=None, timeout=None, session=None):
        mod._response_facts.epoch_verified = False  # this response carried none
        return {"url": "https://other.example/", "title": "Other"}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", side_effect=_post_without_epoch),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs(
            {"snapshot": "- heading \"Other\"", "refsCount": 1},
            url="https://bank.example/login",
        )),
    ):
        json.loads(camofox_navigate("https://other.example/", task_id="agent-task"))

    assert session["privacy_filter_after_handback"] is True


def test_vision_uses_the_privacy_state_from_when_it_captured():
    """A concurrent navigate must not unblock an already-captured screenshot.

    The filter is off when the capture is issued — otherwise the pre-request
    check would stop it and the race would never be reached — this response's
    epoch reveals the handback, and the other turn clears the shared flag
    before the check.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_vision

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _raw(path, params=None, timeout=None, session=None):
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        # The other turn's navigate lands while this capture is in flight.
        session["privacy_filter_after_handback"] = False
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


def test_get_images_uses_the_privacy_state_from_when_it_requested():
    """Same rule for image alt/src, which redaction does not remove."""
    from tools.browser_camofox import camofox_get_images

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": True, "epoch": 4, "_lock": None,
    }

    def _get_racing_navigate(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://10.0.0.5/admin"}]}
        session["privacy_filter_after_handback"] = False
        return {"snapshot": '- image "internal-topology"', "refsCount": 1}

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._get", side_effect=_get_racing_navigate),
    ):
        result = json.loads(camofox_get_images(task_id="agent-task"))

    assert result["success"] is False
    assert "internal-topology" not in json.dumps(result)


def test_eval_uses_the_privacy_state_from_when_it_requested():
    """The last read path that still consulted only the post-response flag.

    A concurrent navigate clearing the shared flag must not release a result
    fetched from the page the human just handed back.
    """
    from tools.browser_tool import _camofox_eval

    import tools.browser_camofox as mod

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _post_racing_navigate(path, body=None, timeout=None, session=None):
        # This response is the one that reveals the handback...
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        # ...and a concurrent navigate clears the shared flag before the check.
        session["privacy_filter_after_handback"] = False
        return {"result": "hunter2"}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._post", side_effect=_post_racing_navigate),
    ):
        result = json.loads(_camofox_eval("document.forms[0].password.value", "agent-task"))

    assert result["success"] is False
    assert "hunter2" not in json.dumps(result)


def test_navigate_inline_snapshot_guard_survives_a_racing_clear():
    """navigate's own snapshot needs the same three-way test as the rest.

    Filter off when the capture is issued, this response reveals the handback,
    a concurrent turn clears the shared flag before the check.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_navigate

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _get_racing(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "http://169.254.169.254/latest/meta-data/"}]}
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        session["privacy_filter_after_handback"] = False
        return {"snapshot": "- text \"iam credentials\"", "refsCount": 1}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/", "title": "Example"}),
        patch("tools.browser_camofox._get", side_effect=_get_racing),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert result.get("snapshot_withheld") is True
    assert "iam credentials" not in json.dumps(result)


def test_vision_annotation_is_redacted_when_only_the_response_reveals_handback():
    """The annotation reaches the vision model, so assert on what it is sent.

    The only signal here is "this response revealed the handback": the filter
    was off when the read was issued and a concurrent turn cleared it again
    before the check. The guard's own /tabs lookup resets that per-response
    fact, so it has to be latched before the guard runs.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_vision

    mod._response_facts.started_handback = False
    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _raw(path, params=None, timeout=None, session=None):
        mod._response_facts.started_handback = False
        resp = MagicMock()
        resp.content = b"\x89PNG"
        return resp

    def _get_racing(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            mod._response_facts.started_handback = False  # the guard's own call
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "https://example.com/"}]}
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        session["privacy_filter_after_handback"] = False  # the other turn clears it
        return {"snapshot": 'textbox "Password" value="hunter2"', "refsCount": 1}

    sent = {}

    def _call_llm(messages=None, **kwargs):
        sent["prompt"] = messages[0]["content"][0]["text"]
        return "ok"

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._get_raw", side_effect=_raw),
        patch("tools.browser_camofox._get", side_effect=_get_racing),
        patch("agent.auxiliary_client.call_llm", side_effect=_call_llm),
    ):
        camofox_vision("what is here?", annotate=True, task_id="agent-task")

    assert "hunter2" not in sent.get("prompt", ""), "the annotation reached the model unredacted"


def test_navigate_snapshot_is_redacted_when_only_the_response_reveals_handback():
    """Same invariant for navigate's inline snapshot."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_navigate

    mod._response_facts.started_handback = False
    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _get_racing(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            mod._response_facts.started_handback = False
            return {"tabs": [{"tabId": "tab-1", "listItemId": "s", "url": "https://example.com/"}]}
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        session["privacy_filter_after_handback"] = False
        return {"snapshot": 'textbox "Password" value="hunter2"', "refsCount": 1}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", return_value={"url": "https://example.com/", "title": "Example"}),
        patch("tools.browser_camofox._get", side_effect=_get_racing),
    ):
        result = json.loads(camofox_navigate("https://example.com/", task_id="agent-task"))

    assert "hunter2" not in json.dumps(result)


    def _post_racing(path, body=None, timeout=None, session=None):
        mod._response_facts.started_handback = True
        session["privacy_filter_after_handback"] = True
        session["privacy_filter_after_handback"] = False
        return {"url": "https://idp.example/callback?code=SECRET"}

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._post", side_effect=_post_racing),
    ):
        result = json.loads(camofox_click("@e1", task_id="agent-task"))

    assert "SECRET" not in json.dumps(result)
    assert result["url"] == "https://idp.example/"


def test_retryable_failure_does_not_erase_a_revealed_handback():
    """An earlier attempt can carry the epoch and still fail retryably."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import _get

    attempts = {"n": 0}
    unavailable = MagicMock()
    unavailable.status_code = 503
    unavailable.headers = {}
    unavailable.json.return_value = {}
    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    ok.json.return_value = {"snapshot": "ready"}

    def _flaky(url, params=None, timeout=None, headers=None, allow_redirects=None, **_kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # This response revealed the handback, then failed retryably.
            mod._response_facts.started_handback = True
            return unavailable
        return ok

    with (
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
        patch("tools.browser_camofox._auth_headers", return_value={}),
        patch("time.sleep", return_value=None),
        patch("tools.browser_camofox.requests.get", side_effect=_flaky),
    ):
        _get("/tabs/tab-1/snapshot")

    assert mod._last_response_started_handback() is True, "the fact must survive the retry"


def test_session_cache_has_a_hard_ceiling():
    """Time-based expiry cannot bound memory on a 2 GB device."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import _evict_surplus_sessions_locked

    mod._sessions.clear()
    for i in range(mod._MAX_TRACKED_SESSIONS + 5):
        mod._sessions[f"p\x00s\x00task-{i:03d}"] = {
            "user_id": "p", "session_key": "s", "task_id": f"task-{i:03d}",
            "managed": True, "local_server_managed": True,
            "last_used_at": float(i),
        }
    evicted = _evict_surplus_sessions_locked()

    assert len(mod._sessions) == mod._MAX_TRACKED_SESSIONS
    assert len(evicted) == 5
    # The least recently used go first, and they come back for teardown rather
    # than being dropped silently.
    assert [s["task_id"] for s in evicted] == [f"task-{i:03d}" for i in range(5)]
    mod._sessions.clear()


def test_a_mutation_cannot_land_inside_another_turns_navigation():
    """Shared tab: a click between a navigate and its snapshot swaps the page.

    The navigate would then report its own url with the other document's
    snapshot and refs, and the Agent's next action lands on the wrong page.
    """
    import threading

    import tools.browser_camofox as mod
    from tools.browser_camofox import camofox_click, camofox_navigate

    mod._owner_locks.clear()
    mod._owner_lock_refs.clear()
    session = {
        "user_id": "profileA", "session_key": "shared", "tab_id": "tab-1",
        "privacy_filter_after_handback": False, "epoch": 1, "_lock": None,
    }
    order = []
    navigate_in_flight = threading.Event()
    click_done = threading.Event()

    def _post(path, body=None, timeout=None, session=None):
        if path.endswith("/navigate"):
            order.append("navigate")
            navigate_in_flight.set()
            # Give the other turn every chance to slip in before the snapshot.
            # Holding the identity lock is what must stop it.
            click_done.wait(0.5)
            return {"url": "https://a.example/", "title": "A"}
        order.append("click")
        click_done.set()
        return {"url": "https://b.example/"}

    def _get(path, params=None, timeout=None, session=None, **kwargs):
        if path == "/tabs":
            return {"tabs": [{"tabId": "tab-1", "listItemId": "shared", "url": "https://a.example/"}]}
        order.append("snapshot")
        return {"snapshot": "- heading \"A\"", "refsCount": 1}

    def _click_turn():
        navigate_in_flight.wait(2)
        with (
            patch("tools.browser_camofox._get_session", return_value=session),
            patch("tools.browser_camofox._post", side_effect=_post),
        ):
            camofox_click("@e1", task_id="turn-b")

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox._session_lock", return_value=MagicMock()),
        patch("tools.browser_camofox._post", side_effect=_post),
        patch("tools.browser_camofox._get", side_effect=_get),
    ):
        other = threading.Thread(target=_click_turn)
        other.start()
        camofox_navigate("https://a.example/", task_id="turn-a")
        other.join(5)

    # The click may run before or after, but never between the navigate and the
    # snapshot that describes where it landed.
    assert order[:2] == ["navigate", "snapshot"], f"a mutation split the navigation: {order}"
    mod._owner_locks.clear()


def test_direct_close_does_not_force_other_profiles_releases():
    """A blanket drain would fire another profile's release early.

    That profile may be inside its quiet window precisely because someone is
    mid-takeover on it.
    """
    import tools.browser_camofox as mod
    from tools.browser_camofox import _queue_pending_teardown, _run_pending_teardowns

    mod._pending_lease_releases.clear()
    with patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None):
        # Due, not merely queued: only the url filter may keep it out.
        _queue_pending_teardown(
            "release", "http://127.0.0.1:9377/x/_zettlab/release", {}, owner="other\x00p",
        )
        _queue_pending_teardown("delete", "http://127.0.0.1:9377/sessions/mine", {})

        ok = MagicMock()
        ok.status_code = 200
        ok.headers = {}
        with (
            patch("tools.browser_camofox.requests.delete", return_value=ok) as mock_delete,
            patch("tools.browser_camofox.requests.post", return_value=ok) as mock_post,
        ):
            _run_pending_teardowns(only_url="http://127.0.0.1:9377/sessions/mine")

    mock_delete.assert_called_once()
    mock_post.assert_not_called()
    assert len(mod._pending_lease_releases) == 1, "the other profile's release stays queued"
    mod._pending_lease_releases.clear()


def test_a_waiting_mutation_does_not_inherit_a_newer_epoch():
    """Refs are only valid for the page they were read from.

    Another turn can advance the epoch while this one waits for the tab, and
    the request header is built from the shared session — so without a check
    the proxy would see a current epoch attached to stale refs and accept them.
    """
    from tools.browser_camofox import camofox_click

    session = {
        "user_id": "u", "tab_id": "tab-1", "session_key": "s",
        "privacy_filter_after_handback": False, "epoch": 4, "_lock": None,
    }

    def _post_should_not_run(path, body=None, timeout=None, session=None):
        raise AssertionError("the mutation must not reach the runtime")

    def _advance_epoch(owner):
        # Stand in for the other turn finishing a navigate while this one waits.
        session["epoch"] = 5
        return _real_lock(owner)

    import tools.browser_camofox as mod
    _real_lock = mod._held_owner_lock

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._post", side_effect=_post_should_not_run),
        patch("tools.browser_camofox._held_owner_lock", side_effect=_advance_epoch),
    ):
        result = json.loads(camofox_click("@e1", task_id="agent-task"))

    assert result["success"] is False
    assert result["error"] == "browser_epoch_stale"
    assert result["retryable"] is True


def test_a_busy_session_is_never_evicted():
    """Evicting mid-operation closes the tab under a running tool call."""
    import tools.browser_camofox as mod
    from tools.browser_camofox import _evict_surplus_sessions_locked

    mod._sessions.clear()
    base = time.monotonic() - 1000
    # One busy entry, oldest of all, plus enough idle ones to exceed the cap.
    mod._sessions["p\x00s\x00busy"] = {
        "user_id": "p", "session_key": "s", "task_id": "busy",
        "in_flight": 1, "last_used_at": base - 100,
    }
    for i in range(mod._MAX_TRACKED_SESSIONS + 2):
        mod._sessions[f"p\x00s\x00idle-{i:03d}"] = {
            "user_id": "p", "session_key": "s", "task_id": f"idle-{i:03d}",
            "in_flight": 0, "last_used_at": base + i,
        }

    evicted = _evict_surplus_sessions_locked()

    assert "p\x00s\x00busy" in mod._sessions, "a session with work in flight was evicted"
    assert all(s["task_id"] != "busy" for s in evicted)
    # Recency alone must not spare an idle entry: the ceiling is hard, and it is
    # reached by evicting idle entries rather than by sparing recent ones.
    assert len(mod._sessions) == mod._MAX_TRACKED_SESSIONS, "the ceiling was not enforced"
    mod._sessions.clear()


def test_a_new_session_is_never_the_one_evicted():
    """The entry this call is about to use is exempt from capacity eviction.

    Its ``in_flight`` is still 0 when the ceiling is enforced — the reference is
    taken by the tool call, after ``_get_session`` returns. With every other
    entry busy it would be the only evictable one, and the caller would receive
    a session that is no longer tracked: direct mode would DELETE a tab it has
    not created yet and then lose the one it does create; managed mode would
    schedule the release of a runtime that is still in use.
    """
    import tools.browser_camofox as mod

    mod._sessions.clear()
    base = time.monotonic() - 1000
    for i in range(mod._MAX_TRACKED_SESSIONS):
        mod._sessions[f"p\x00s\x00busy-{i:03d}"] = {
            "user_id": "u", "session_key": "s", "task_id": f"busy-{i:03d}",
            "in_flight": 1, "last_used_at": base + i,
        }
    fresh_key = "p\x00s\x00fresh"
    mod._sessions[fresh_key] = {
        "user_id": "u", "session_key": "s", "task_id": "fresh",
        "in_flight": 0, "last_used_at": base + 10_000,
    }

    evicted = mod._evict_surplus_sessions_locked(protect_key=fresh_key)

    assert fresh_key in mod._sessions, "the session about to be returned was evicted"
    assert all(s.get("task_id") != "fresh" for s in evicted)
    mod._sessions.clear()


def test_handback_reads_reapply_the_website_policy():
    """A page reached by human takeover must clear the same policy as a navigate.

    security.website_blocklist is enforced in browser_navigate(); without it
    here, a human could hand back a blocked site and its content would still go
    to the model through snapshot, get_images or vision.
    """
    import tools.browser_camofox as mod

    with (
        patch("tools.browser_tool._is_always_blocked_url", return_value=False),
        patch("tools.browser_tool._is_safe_url", return_value=True),
    ):
        with patch("tools.website_policy.check_website_access", return_value=None):
            assert mod._recovery_target_allowed("https://allowed.example/page")
        with patch("tools.website_policy.check_website_access", return_value="blocked by policy"):
            assert not mod._recovery_target_allowed("https://blocked.example/page")
    # The SSRF guards still run first and independently of the policy.
    with patch("tools.website_policy.check_website_access", return_value=None):
        assert not mod._recovery_target_allowed("http://169.254.169.254/latest/meta-data/")


def test_a_tab_rebound_mid_navigation_is_also_held():
    """A 404 mid-navigation replaces the session; the new one needs a reference.

    The caller's _session_operation holds the entry the call started with, so
    without this the replacement is the one entry capacity eviction can drop —
    while the navigation it was created for is still running.
    """
    import tools.browser_camofox as mod

    original = {"user_id": "u", "session_key": "s", "tab_id": "tab-old", "task_id": "t",
                "epoch": None, "in_flight": 0, "last_used_at": time.monotonic()}
    # _ensure_tab goes through _get_session, which hands back an entry that
    # already holds its reference — the replacement arrives referenced and this
    # call has to release it.
    replacement = {"user_id": "u", "session_key": "s", "tab_id": "tab-new", "task_id": "t",
                   "epoch": None, "in_flight": 1, "last_used_at": time.monotonic()}
    in_flight_during_navigate = {}

    gone = requests.HTTPError()
    gone.response = MagicMock(status_code=404)

    def _post(path, body=None, timeout=None, session=None):
        if "tab-old" in path:
            raise gone
        in_flight_during_navigate["value"] = int(replacement.get("in_flight") or 0)
        return {"url": "https://a.example/"}

    with (
        patch("tools.browser_camofox._ensure_tab", return_value=replacement),
        patch("tools.browser_camofox._post", side_effect=_post),
        patch("tools.browser_camofox._get", side_effect=requests.HTTPError()),
    ):
        mod._navigate_within_identity(original, "t", "https://a.example/", "https://a.example/", None)

    assert in_flight_during_navigate.get("value", 0) > 0, "the rebound tab was navigated with no reference held"
    assert int(replacement.get("in_flight") or 0) == 0, "the reference was never released"


def test_no_path_out_of_get_session_leaks_a_reference():
    """Every exit has to give the reference back, refusals and failures too.

    A leaked reference is permanent — the entry is skipped by the idle sweep and
    by eviction — so enough of them make every later call fail with
    browser_sessions_busy and strand the tabs behind them.
    """
    import tools.browser_camofox as mod
    from tools.browser_tool import _camofox_eval

def test_a_refused_navigation_target_reaches_the_agent_verbatim():
    """local-server refuses targets the Agent may not browse to; it must learn why.

    The proxy answers 403 browser_target_not_allowed for a non-http scheme or a
    private address. A generic "navigation failed" would have the model retry
    the same URL; the reason and the fact that it is not retryable both have to
    survive the error filter.
    """
    import tools.browser_camofox as mod

    session = {"user_id": "u", "session_key": "s", "tab_id": "tab-1", "task_id": "t", "epoch": 2}
    refused = MagicMock(status_code=403)
    refused.json.return_value = {
        "error": "browser_target_not_allowed",
        "message": "only http and https browser targets are allowed",
    }

    with (
        patch("tools.browser_camofox._get_session", return_value=session),
        patch("tools.browser_camofox._ensure_tab", return_value=session),
        patch("tools.browser_camofox.requests.post", return_value=refused),
    ):
        result = json.loads(mod.camofox_navigate("file:///etc/shadow", task_id="t"))

    assert result["success"] is False
    assert result["error"] == "browser_target_not_allowed"
    assert "http and https" in result["message"]
    assert result.get("retryable") is not True, "a policy refusal must not be advertised as retryable"


def test_action_token_requests_never_use_an_environment_proxy(monkeypatch):
    """Validating the URL is not enough; the transport has to refuse the proxy.

    requests honours HTTP_PROXY / ALL_PROXY whenever NO_PROXY does not cover
    loopback, so a device with a proxy configured would hand this device's Agent
    authority to whatever host that proxy points at.
    """
    import tools.browser_camofox as mod

    monkeypatch.setenv("HTTP_PROXY", "http://attacker.example:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.example:3128")
    monkeypatch.setenv("ALL_PROXY", "http://attacker.example:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)

    session = {"user_id": "u", "session_key": "s", "tab_id": "tab-1", "task_id": "t", "epoch": 2}
    seen = {}

    def _record(url, **kwargs):
        seen[url] = kwargs.get("proxies")
        response = MagicMock(status_code=200)
        response.json.return_value = {"ok": True}
        response.headers = {}
        return response

    with (
        patch("tools.browser_camofox.requests.post", side_effect=_record),
        patch("tools.browser_camofox.requests.get", side_effect=_record),
        patch("tools.browser_camofox.requests.delete", side_effect=_record),
    ):
        mod._post("/tabs/tab-1/click", {"userId": "u"}, session=session)
        mod._get("/tabs/tab-1/snapshot", params={"userId": "u"}, session=session)
        mod._delete("/tabs/tab-1", {"userId": "u"}, session=session)

    assert seen, "no request was made"
    for url, proxies in seen.items():
        assert proxies is not None, f"{url} inherited the environment proxy"
        for scheme in ("http", "https", "all"):
            assert proxies.get(scheme, "unset") is None, f"{url} left {scheme} proxying enabled"


def test_two_profiles_do_not_share_remembered_tab_epochs():
    """The epoch memory is the last per-process record keyed without the owner.

    Two profiles handed the same explicit identity can be given the same tab id
    by their own runtimes. Sharing this record would let one profile's
    "filter was off" become the other's trusted state on adoption, clearing a
    live handback filter over a page the other user just typed into.
    """
    import tools.browser_camofox as mod

    a = {"user_id": "shared", "session_key": "sess", "release_owner": "shared\x00digest-a"}
    b = {"user_id": "shared", "session_key": "sess", "release_owner": "shared\x00digest-b"}
    assert mod._tab_epoch_memory_key(a, "tab-1") != mod._tab_epoch_memory_key(b, "tab-1")

    with mod._sessions_lock:
        mod._remembered_tab_epochs.clear()
        mod._remembered_tab_epochs[mod._tab_epoch_memory_key(b, "tab-1")] = (7, True)
        mod._remembered_tab_epochs[mod._tab_epoch_memory_key(a, "tab-1")] = (7, False)

    assert mod._remembered_tab_state(b, "tab-1") == (7, True), "profile A overwrote profile B's privacy state"
    assert mod._remembered_tab_state(a, "tab-1") == (7, False)
    with mod._sessions_lock:
        mod._remembered_tab_epochs.clear()


def test_a_refusal_never_hands_over_a_usable_epoch(managed_session):
    """The barrier is only worth what the client cannot shortcut.

    local-server no longer puts the current epoch in a browser_epoch_stale
    body, but an older or hostile proxy still can. Adopting it would clear the
    barrier without the snapshot that carries the human's page state, so this
    side must ignore it whether the recovery snapshot succeeds or fails.
    """
    import tools.browser_camofox as mod

    before = managed_session["epoch"]
    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(99)),
        patch("tools.browser_camofox._get", side_effect=requests.ConnectionError("boom")),
    ):
        json.loads(camofox_click("@e4", task_id="agent-task"))
    assert managed_session["epoch"] == before, "the refusal's epoch was adopted"

    with (
        patch("tools.browser_camofox._get_session", return_value=managed_session),
        patch("tools.browser_camofox._post", side_effect=_epoch_stale(99)),
        patch("tools.browser_camofox._get", side_effect=_get_serving_tabs({"snapshot": "- button [e9]", "refsCount": 1})),
    ):
        json.loads(camofox_click("@e4", task_id="agent-task"))
    assert managed_session["epoch"] == before, "the refusal's epoch was adopted after a successful re-snapshot"

    # Only a response carrying the header moves it.
    response = MagicMock(status_code=200, headers={mod._EPOCH_HEADER: "7"})
    mod._adopt_epoch_from_response(managed_session, response, tab_operation=True)
    assert managed_session["epoch"] == 7


def test_a_full_session_cache_applies_backpressure():
    """A cache full of work in flight must refuse, not grow.

    Protecting only the entry being returned makes the ceiling soft again: with
    every existing entry busy, each new key protects itself and the cache — and
    the browser tabs behind it — grows with the number of concurrent sessions.
    """
    import tools.browser_camofox as mod

    mod._sessions.clear()
    base = time.monotonic()
    for i in range(mod._MAX_TRACKED_SESSIONS):
        mod._sessions[f"o\x00u\x00s\x00busy-{i:03d}"] = {
            "user_id": "u", "session_key": "s", "task_id": f"busy-{i:03d}",
            "in_flight": 1, "last_used_at": base + i,
        }

    with (
        patch("tools.browser_camofox._get_camofox_config", return_value={}),
        patch("tools.browser_camofox._camofox_identity_override", return_value=None),
        patch("tools.browser_camofox.get_camofox_identity", return_value={"user_id": "u", "session_key": "s"}),
        patch("tools.browser_camofox._release_owner_key", return_value="o"),
        patch("tools.browser_camofox._local_server_managed", return_value=False),
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
        patch("tools.browser_camofox._auth_headers", return_value={}),
        patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None),
    ):
        with pytest.raises(mod.CamofoxSessionsBusy):
            mod._get_session("newcomer")

    # The refused entry left nothing behind.
    assert len(mod._sessions) == mod._MAX_TRACKED_SESSIONS
    assert all(not k.endswith("newcomer") for k in mod._sessions)
    mod._sessions.clear()


def test_backpressure_never_turns_away_an_existing_session():
    """Refusing is for admitting new work, not for work already tracked.

    A turn coming back to a session the cache already holds adds nothing to it;
    turning that call away because its neighbours are busy would break a session
    that is behaving.
    """
    import tools.browser_camofox as mod

    mod._sessions.clear()
    base = time.monotonic()
    identity = {"user_id": "u", "session_key": "s"}
    for i in range(mod._MAX_TRACKED_SESSIONS - 1):
        mod._sessions[f"o\x00u\x00s\x00busy-{i:03d}"] = {
            "user_id": "u", "session_key": "s", "task_id": f"busy-{i:03d}",
            "in_flight": 1, "last_used_at": base + i,
        }
    mine = mod._session_cache_key("mine", identity, "o")
    mod._sessions[mine] = {
        "user_id": "u", "session_key": "s", "task_id": "mine", "tab_id": "tab-1",
        "in_flight": 1, "last_used_at": base,
    }

    with (
        patch("tools.browser_camofox._get_camofox_config", return_value={}),
        patch("tools.browser_camofox._camofox_identity_override", return_value=None),
        patch("tools.browser_camofox.get_camofox_identity", return_value=identity),
        patch("tools.browser_camofox._release_owner_key", return_value="o"),
        patch("tools.browser_camofox._local_server_managed", return_value=False),
        patch("tools.browser_camofox.get_camofox_url", return_value="http://127.0.0.1:9377"),
        patch("tools.browser_camofox._auth_headers", return_value={}),
        patch("tools.browser_camofox._ensure_maintenance_worker", return_value=None),
        patch("tools.browser_camofox._adopt_existing_tab", side_effect=lambda s: s),
    ):
        session = mod._get_session("mine")
    assert session is mod._sessions[mine]
    mod._end_session_call(session)
    mod._sessions.clear()


def test_the_capture_guard_does_not_depend_on_the_filter_state():
    """The first post-handback capture is the one that turns the filter on.

    Deciding whether to hold the lock by the state before the request therefore
    leaves exactly that capture unprotected, and a concurrent navigate can move
    the tab to an allowed page before the readability check looks.
    """
    import threading

    import tools.browser_camofox as mod

    mod._owner_locks.clear()
    mod._owner_lock_refs.clear()
    session = {"user_id": "u", "session_key": "s", "tab_id": "tab-1", "task_id": "t",
               "privacy_filter_after_handback": False, "epoch": 2}

    held = threading.Event()
    other_turn_entered = threading.Event()

    def _other_turn():
        held.wait(1)
        with mod._held_owner_lock(mod._browser_identity_key(session)):
            other_turn_entered.set()

    worker = threading.Thread(target=_other_turn)
    worker.start()
    with mod._capture_guard(session):
        held.set()
        # The other turn must not be able to take the identity while this
        # capture is judged, even though the filter was off on entry.
        assert not other_turn_entered.wait(0.2), "the capture ran without the tab identity held"
    worker.join(2)
    assert other_turn_entered.is_set()


def test_only_a_successful_response_moves_the_epoch():
    """Adoption runs before _raise_for_status, so a refusal reaches it too.

    An older or hostile proxy putting an epoch header on its own 409 would
    otherwise hand the Agent the value that refusal exists to withhold — and if
    the recovery snapshot then failed, the next press or back would carry it.
    """
    import tools.browser_camofox as mod

    session = {"epoch": 2, "local_server_managed": True, "privacy_filter_after_handback": False}
    refusal = MagicMock(status_code=409, headers={mod._EPOCH_HEADER: "9"})
    mod._adopt_epoch_from_response(session, refusal, tab_operation=True)
    assert session["epoch"] == 2, "a refusal's epoch header was adopted"

    for status in (400, 403, 500, 503):
        session["epoch"] = 2
        mod._adopt_epoch_from_response(
            session, MagicMock(status_code=status, headers={mod._EPOCH_HEADER: "9"}), tab_operation=True
        )
        assert session["epoch"] == 2, f"a {status} epoch header was adopted"

    ok = MagicMock(status_code=200, headers={mod._EPOCH_HEADER: "9"})
    mod._adopt_epoch_from_response(session, ok, tab_operation=True)
    assert session["epoch"] == 9


def test_direct_teardown_is_scoped_to_its_credential():
    """Two profiles can share a URL and a user id and differ only by credential.

    Merging their DELETEs would overwrite one profile's only credential, and a
    targeted drain would fire the other profile's teardown as its own.
    """
    import tools.browser_camofox as mod

    mod._pending_lease_releases.clear()
    url = "http://127.0.0.1:9377/sessions/shared_user"
    mod._queue_pending_teardown("delete", url, {"Authorization": "Bearer a"}, owner="shared\x00digest-a")
    mod._queue_pending_teardown("delete", url, {"Authorization": "Bearer b"}, owner="shared\x00digest-b")

    assert len(mod._pending_lease_releases) == 2, "two profiles' teardowns were merged into one"
    creds = {e["headers"]["Authorization"] for e in mod._pending_lease_releases}
    assert creds == {"Bearer a", "Bearer b"}, "one profile's credential was overwritten"

    with patch("tools.browser_camofox._attempt_teardown", return_value=True) as attempt:
        mod._run_pending_teardowns(only_url=url, only_owner="shared\x00digest-a")
    assert attempt.call_count == 1, "a targeted drain fired another profile's teardown"
    assert [e.get("owner") for e in mod._pending_lease_releases] == ["shared\x00digest-b"]

    # And the queueing path itself has to carry the owner, not just this test.
    mod._pending_lease_releases.clear()
    for tag in ("a", "b"):
        mod._teardown_session({
            "user_id": "shared_user", "session_key": "s", "task_id": f"t-{tag}",
            "managed": False, "local_server_managed": False,
            "delete_base": "http://127.0.0.1:9377",
            "delete_headers": {"Authorization": f"Bearer {tag}"},
            "release_owner": f"shared\x00digest-{tag}",
        })
    assert len(mod._pending_lease_releases) == 2, "_teardown_session merged two profiles into one delete"
    assert {e["headers"]["Authorization"] for e in mod._pending_lease_releases} == {"Bearer a", "Bearer b"}
    mod._pending_lease_releases.clear()


def test_shutdown_releases_every_tracked_session():
    """A daemon worker is killed at interpreter exit, so nothing else would."""
    import tools.browser_camofox as mod

    mod._sessions.clear()
    mod._pending_lease_releases.clear()
    for tag in ("a", "b"):
        mod._sessions[f"o\x00u\x00s\x00{tag}"] = {
            "user_id": f"user-{tag}", "session_key": "s", "task_id": tag, "tab_id": f"tab-{tag}",
            "managed": False, "local_server_managed": False,
            "delete_base": "http://127.0.0.1:9377",
            "delete_headers": {"Authorization": f"Bearer {tag}"},
            "release_owner": f"o-{tag}",
            "in_flight": 0, "last_used_at": time.monotonic(),
        }

    attempted = []
    with patch("tools.browser_camofox._attempt_teardown", side_effect=lambda e: attempted.append(e["url"]) or True):
        mod.shutdown_camofox_sessions()

    assert mod._sessions == {}, "sessions survived shutdown"
    assert len(attempted) == 2, f"not every session was released: {attempted}"
    assert mod._pending_lease_releases == []

