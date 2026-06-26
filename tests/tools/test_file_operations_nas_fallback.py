"""Tests for ShellFileOperations NAS agent-search fallback (ZET-1610).

Covers the four quadrants (success / empty / error / non-Zettlab) plus the
review regressions: malformed reply must degrade to None (not a tool error),
the action token must only reach the registry-injected loopback URL, and the
fallback must not force the semantic mode.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.file_operations import ShellFileOperations, SearchResult


_APPEND_URL = "http://127.0.0.1:9090/api/v1/internal/chat/append"
_EXPECTED_URL = "http://127.0.0.1:9090/api/v1/file/index/agent-search"


@pytest.fixture()
def file_ops():
    return ShellFileOperations(MagicMock())


def _fake_urlopen(payload, captured=None):
    """Return a urlopen replacement that yields `payload` and records the request."""
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode("utf-8")

    def _open(req, timeout=None):
        if captured is not None:
            captured["req"] = req
            captured["timeout"] = timeout
        return FakeResp()

    return _open


def _zettlab_env(monkeypatch):
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", _APPEND_URL)


# --- URL derivation (C1) ----------------------------------------------------

def test_agent_search_url_derived_from_append_url(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", _APPEND_URL)
    assert ShellFileOperations._zettlab_agent_search_url() == _EXPECTED_URL


def test_agent_search_url_none_when_env_missing(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    assert ShellFileOperations._zettlab_agent_search_url() is None


def test_agent_search_url_none_when_malformed(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "not-a-url")
    assert ShellFileOperations._zettlab_agent_search_url() is None


# --- success ----------------------------------------------------------------

def test_fallback_success_returns_note_not_files(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}, {"filename": "b.txt"}],
                        "total_count": 2}}
    with patch("urllib.request.urlopen", _fake_urlopen(payload, captured)):
        result = file_ops._zettlab_nas_fallback("report", 50)

    assert isinstance(result, SearchResult)
    assert result.total_count == 2
    assert result.note and "do not list" in result.note.lower()
    # B3: raw paths are NOT returned — the cards are the sole presentation.
    assert result.files == []
    d = result.to_dict()
    assert "files" not in d
    assert d["note"] == result.note

    # C1: request goes only to the derived loopback URL, with the token.
    req = captured["req"]
    assert req.full_url == _EXPECTED_URL
    assert req.headers.get("X-zettlab-agent-action-token") == "tok"
    # C2: semantic is not forced.
    sent = json.loads(req.data.decode("utf-8"))
    assert sent["modes"] == ["name", "content"]
    assert captured["timeout"] == 10


# --- turn_id header (D1) -----------------------------------------------------

def test_fallback_sends_turn_id_header_when_present(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    from tools.file_operations import set_zettlab_turn_id

    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}], "total_count": 1}}
    # The api_server handler pins this turn's metadata.turn_id onto the feature
    # contextvar; the fallback must echo it back as X-Zettlab-Turn-Id so
    # local-server injects the card into THIS exact turn (ByTurnIDForAgent).
    set_zettlab_turn_id("t_abc-123")
    try:
        with patch("urllib.request.urlopen", _fake_urlopen(payload, captured)):
            result = file_ops._zettlab_nas_fallback("report", 50)
    finally:
        set_zettlab_turn_id("")

    assert result is not None
    # urllib.request.Request stores header names .capitalize()-folded.
    assert captured["req"].headers.get("X-zettlab-turn-id") == "t_abc-123"


def test_fallback_omits_turn_id_header_when_absent(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    from tools.file_operations import set_zettlab_turn_id

    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}], "total_count": 1}}
    # No turn_id this turn (local-server sent none): header absent, not empty.
    set_zettlab_turn_id("")
    try:
        with patch("urllib.request.urlopen", _fake_urlopen(payload, captured)):
            file_ops._zettlab_nas_fallback("report", 50)
    finally:
        set_zettlab_turn_id("")

    assert "X-zettlab-turn-id" not in captured["req"].headers


# --- empty ------------------------------------------------------------------

def test_fallback_empty_items_returns_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    with patch("urllib.request.urlopen", _fake_urlopen({"data": {"items": []}})):
        assert file_ops._zettlab_nas_fallback("nothing", 50) is None


# --- error / malformed (B1 regression) --------------------------------------

@pytest.mark.parametrize("payload", [
    "just a string",          # json.loads -> str, .get would raise AttributeError
    [1, 2, 3],                # json.loads -> list
    None,                     # json.loads -> None
    {"data": "oops"},         # data not a dict
    {"data": {"items": "x"}},  # items not iterable of dicts
    {"data": {"items": [42, "str"]}},  # items contains non-dicts -> 0 hits
    {"data": {"items": [{"path": "/a"}], "total_count": "NaN"}},  # bad total_count
])
def test_fallback_malformed_payload_returns_none(monkeypatch, file_ops, payload):
    _zettlab_env(monkeypatch)
    with patch("urllib.request.urlopen", _fake_urlopen(payload)):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


def test_fallback_network_error_returns_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)

    def _boom(req, timeout=None):
        raise OSError("connection refused")

    with patch("urllib.request.urlopen", _boom):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


# --- non-Zettlab (no token / no callback URL) -------------------------------

def test_fallback_no_token_returns_none(monkeypatch, file_ops):
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", _APPEND_URL)
    # Must not even attempt the request without a token.
    with patch("urllib.request.urlopen", side_effect=AssertionError("should not call")):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


def test_fallback_no_callback_url_returns_none(monkeypatch, file_ops):
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    with patch("urllib.request.urlopen", side_effect=AssertionError("should not call")):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


# --- search() wrapper gating ------------------------------------------------

def test_search_returns_workspace_hit_without_fallback(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    hit = SearchResult(files=["local.txt"], total_count=1)
    with patch.object(file_ops, "_search_workspace", return_value=hit), \
         patch.object(file_ops, "_zettlab_nas_fallback") as fb:
        out = file_ops.search("x")
    assert out is hit
    fb.assert_not_called()  # workspace had results -> no NAS round-trip


def test_search_falls_back_when_workspace_empty(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    empty = SearchResult(total_count=0)
    nas = SearchResult(total_count=3, note="cards rendered")
    with patch.object(file_ops, "_search_workspace", return_value=empty), \
         patch.object(file_ops, "_zettlab_nas_fallback", return_value=nas):
        out = file_ops.search("x")
    assert out is nas


def test_search_returns_empty_when_fallback_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    empty = SearchResult(total_count=0)
    with patch.object(file_ops, "_search_workspace", return_value=empty), \
         patch.object(file_ops, "_zettlab_nas_fallback", return_value=None):
        out = file_ops.search("x")
    assert out is empty  # graceful: valid empty search, not an error


def test_search_skips_fallback_on_workspace_error(monkeypatch, file_ops):
    # D2: an errored workspace search (path not found, bad regex, rg/grep hard
    # failure) also has total_count == 0, but must surface the error — not be
    # masked by an unrelated NAS hit.
    _zettlab_env(monkeypatch)
    errored = SearchResult(error="Path not found: /missing", total_count=0)
    with patch.object(file_ops, "_search_workspace", return_value=errored), \
         patch.object(file_ops, "_zettlab_nas_fallback") as fb:
        out = file_ops.search("x", path="/missing")
    assert out is errored
    fb.assert_not_called()
