"""Focused coverage for the profile-scoped device meeting bridge."""

import json
from urllib.error import HTTPError
from unittest.mock import patch

from tools.device_meetings_tool import _available, device_meetings_tool


def test_device_meetings_requires_profile_scope_in_multiplex(monkeypatch):
    from agent import secret_scope

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        assert _available() is False
        assert json.loads(device_meetings_tool({"action": "list"})) == {
            "error": {
                "code": "unavailable",
                "message": "Device meeting bridge is unavailable.",
            }
        }
    finally:
        secret_scope.set_multiplex_active(previous)


def test_device_meetings_profile_scope_flow_uses_scoped_credentials(monkeypatch):
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return b'{"meetings":[]}'

    def open_(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        assert _available() is True
        with patch("urllib.request.urlopen", open_):
            result = device_meetings_tool({"action": "list", "limit": 1, "offset": 0})

    assert json.loads(result) == {"meetings": []}
    assert seen["request"].full_url == (
        "http://127.0.0.1:9420/api/v1/internal/meetings?limit=1&offset=0"
    )
    assert seen["request"].get_header("X-zettlab-agent-action-token") == "profile-token"
    assert seen["timeout"] == 8
    assert getattr(_available, "_profile_scope_sensitive") is True


def test_device_meetings_unwraps_local_server_envelope_and_bounds_args(monkeypatch):
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/ignored",
        "ZETTLAB_AGENT_ACTION_TOKEN": "scoped-token",
    }
    seen = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            seen["read_size"] = size
            return b'{"code":200,"data":{"meetings":[{"id":"m-1"}]}}'

    def open_(request, timeout=None):
        seen["url"] = request.full_url
        return Response()

    with mux_profile_scope(monkeypatch, scope):
        with patch("urllib.request.urlopen", open_):
            result = device_meetings_tool({"action": "list", "limit": 999, "offset": -4})

    assert json.loads(result) == {"meetings": [{"id": "m-1"}]}
    assert seen["url"].endswith("/api/v1/internal/meetings?limit=20&offset=0")
    assert seen["read_size"] == 512 * 1024 + 1


def test_device_meetings_clamps_offset_to_bridge_limit(monkeypatch):
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420",
        "ZETTLAB_AGENT_ACTION_TOKEN": "scoped-token",
    }
    seen = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b'{"meetings":[]}'

    def open_(request, timeout=None):
        seen["url"] = request.full_url
        return Response()

    with mux_profile_scope(monkeypatch, scope):
        with patch("urllib.request.urlopen", open_):
            device_meetings_tool({"action": "list", "offset": 999999})

    assert seen["url"].endswith("/api/v1/internal/meetings?limit=20&offset=100000")


def test_device_meetings_failures_are_stable_and_retry_transient_http(monkeypatch):
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420",
        "ZETTLAB_AGENT_ACTION_TOKEN": "scoped-token",
    }
    calls = {"count": 0}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b"not-json"

    def open_(request, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(request.full_url, 503, "busy", {}, None)
        return Response()

    with mux_profile_scope(monkeypatch, scope):
        with patch("urllib.request.urlopen", open_):
            result = device_meetings_tool({"action": "list"})

    assert calls["count"] == 2
    assert json.loads(result) == {
        "error": {
            "code": "invalid_response",
            "message": "Device meeting bridge returned an invalid response.",
        }
    }


def test_device_meetings_rejects_credentials_in_base_url(monkeypatch):
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": "http://user:pass@127.0.0.1:9420",
        "ZETTLAB_AGENT_ACTION_TOKEN": "scoped-token",
    }
    with mux_profile_scope(monkeypatch, scope):
        assert _available() is False
        assert json.loads(device_meetings_tool({"action": "list"}))["error"]["code"] == "unavailable"
