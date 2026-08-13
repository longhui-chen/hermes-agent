"""Focused coverage for the profile-scoped device meeting bridge."""

import json
from unittest.mock import patch

from tools.device_meetings_tool import _available, device_meetings_tool


def test_device_meetings_requires_profile_scope_in_multiplex(monkeypatch):
    from agent import secret_scope

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        assert _available() is False
        assert json.loads(device_meetings_tool({"action": "list"})) == {
            "error": "device meeting bridge unavailable"
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
