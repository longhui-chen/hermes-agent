"""Regression test: _ensure_tab must send ``listItemId`` (not ``sessionKey``).

The Camoufox REST API server requires ``listItemId`` in the ``POST /tabs``
body.  A previous version sent ``sessionKey`` which caused a 400 Bad Request
on every ``browser_navigate`` call.  See issue #37960.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch, MagicMock


def test_ensure_tab_sends_list_item_id():
    """POST /tabs body must contain ``listItemId``, not ``sessionKey``."""
    # Import the module under test
    from tools import browser_camofox as mod

    fake_session = {
        "user_id": "hermes_test123",
        "tab_id": None,
        "session_key": "task_my-session",
        "managed": False,
        "adopt_existing_tab": False,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tabId": "tab-42"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(mod, "_get_session", return_value=fake_session), \
         patch.object(mod, "_get_command_timeout", return_value=30), \
         patch.object(mod, "get_camofox_url", return_value="http://localhost:9377"), \
         patch("tools.browser_camofox.requests.post", return_value=mock_response) as mock_post:
        result = mod._ensure_tab("test-task", url="https://example.com")

    # Verify the POST was called
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

    # Core assertion: listItemId present, sessionKey absent
    assert "listItemId" in body, f"Expected 'listItemId' in POST body, got: {body}"
    assert "sessionKey" not in body, f"'sessionKey' should not be in POST body: {body}"
    assert body["listItemId"] == "task_my-session"
    assert body["userId"] == "hermes_test123"
    assert body["url"] == "https://example.com"
    assert call_kwargs.kwargs["timeout"] == 60

    # Verify tab_id was set from response
    assert result["tab_id"] == "tab-42"


def test_ensure_tab_omits_url_for_blank_tab():
    """Camofox 1.13 creates a blank tab only when ``url`` is absent."""
    from tools import browser_camofox as mod

    fake_session = {
        "user_id": "hermes_test123",
        "tab_id": None,
        "session_key": "task_my-session",
        "managed": False,
        "adopt_existing_tab": False,
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"tabId": "tab-blank"}

    with patch.object(mod, "_get_session", return_value=fake_session), \
         patch("tools.browser_camofox.requests.post", return_value=mock_response) as mock_post:
        result = mod._ensure_tab("test-task")

    assert "url" not in mock_post.call_args.kwargs["json"]
    assert result["tab_id"] == "tab-blank"


def test_ensure_tab_skips_creation_when_tab_exists():
    """If session already has a tab_id, no POST should be made."""
    from tools import browser_camofox as mod

    fake_session = {
        "user_id": "hermes_test123",
        "tab_id": "existing-tab",
        "session_key": "task_my-session",
        "managed": False,
    }

    with patch.object(mod, "_get_session", return_value=fake_session), \
         patch("tools.browser_camofox.requests.post") as mock_post:
        result = mod._ensure_tab("test-task")

    # No POST should be made — tab already exists
    mock_post.assert_not_called()
    assert result["tab_id"] == "existing-tab"


def test_ensure_tab_singleflights_concurrent_creation():
    from tools import browser_camofox as mod

    session = {
        "user_id": "profile-1", "tab_id": None, "session_key": "task-1",
        "managed": True, "adopt_existing_tab": False,
    }
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"tabId": "tab-only"}
    response.raise_for_status = MagicMock()

    with patch.object(mod, "_get_session", return_value=session), \
         patch.object(mod, "get_camofox_url", return_value="http://localhost:9377"), \
         patch("tools.browser_camofox.requests.post", return_value=response) as mock_post, \
         ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: mod._ensure_tab("task-1"), range(2)))

    assert mock_post.call_count == 1
    assert [result["tab_id"] for result in results] == ["tab-only", "tab-only"]


def test_slow_profile_does_not_block_another_profile():
    from tools import browser_camofox as mod

    sessions = {
        "slow": {"user_id": "slow", "tab_id": None, "session_key": "slow", "managed": True, "adopt_existing_tab": False},
        "fast": {"user_id": "fast", "tab_id": None, "session_key": "fast", "managed": True, "adopt_existing_tab": False},
    }
    slow_started = Event()
    release_slow = Event()

    def create(_url, json, **_kwargs):
        if json["userId"] == "slow":
            slow_started.set()
            assert release_slow.wait(1)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"tabId": f"tab-{json['userId']}"}
        response.raise_for_status = MagicMock()
        return response

    with patch.object(mod, "_get_session", side_effect=lambda task: sessions[task]), \
         patch.object(mod, "get_camofox_url", return_value="http://localhost:9377"), \
         patch("tools.browser_camofox.requests.post", side_effect=create), \
         ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(mod._ensure_tab, "slow")
        assert slow_started.wait(1)
        fast = pool.submit(mod._ensure_tab, "fast")
        assert fast.result(timeout=0.5)["tab_id"] == "tab-fast"
        release_slow.set()
        assert slow.result(timeout=1)["tab_id"] == "tab-slow"
