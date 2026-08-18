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
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
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
        with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
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
        with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
            file_ops._zettlab_nas_fallback("report", 50)
    finally:
        set_zettlab_turn_id("")

    assert "X-zettlab-turn-id" not in captured["req"].headers


# --- empty ------------------------------------------------------------------

def test_fallback_empty_items_returns_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen({"data": {"items": []}})):
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
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


def test_fallback_network_error_returns_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)

    def _boom(req, timeout=None):
        raise OSError("connection refused")

    with patch("tools.file_operations.urlopen_hardened", _boom):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


# --- non-Zettlab (no token / no callback URL) -------------------------------

def test_fallback_no_token_returns_none(monkeypatch, file_ops):
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", _APPEND_URL)
    # Must not even attempt the request without a token.
    with patch("tools.file_operations.urlopen_hardened", side_effect=AssertionError("should not call")):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


def test_fallback_no_callback_url_returns_none(monkeypatch, file_ops):
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    with patch("tools.file_operations.urlopen_hardened", side_effect=AssertionError("should not call")):
        assert file_ops._zettlab_nas_fallback("q", 50) is None


# --- shared-gateway profile scope flow ---------------------------------------

def test_profile_scope_flow_fallback_works_with_poisoned_environ(monkeypatch, file_ops):
    """Shared gateway mode: token + callback URL live only in the profile
    secret scope while os.environ holds another profile's stale decoys — the
    NAS request must be built entirely from the scope."""
    from tests.tools._profile_scope import mux_profile_scope, request_fingerprint

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}], "total_count": 1}}
    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
            result = file_ops._zettlab_nas_fallback("report", 50)

    assert result is not None and result.total_count == 1
    req = captured["req"]
    assert req.full_url == "http://127.0.0.1:9420/api/v1/file/index/agent-search"
    assert req.headers.get("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)


def test_profile_scope_flow_search_gate_reads_scope(monkeypatch, file_ops):
    """search() must decide 'is this a Zettlab device' from the profile scope,
    not from os.environ (empty here)."""
    from tests.tools._profile_scope import mux_profile_scope

    scope = {
        "ZET_CHAT_APPEND_URL": _APPEND_URL,
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    empty = SearchResult(total_count=0)
    nas = SearchResult(total_count=3, note="cards rendered")
    with mux_profile_scope(monkeypatch, scope):  # scope keys purged from env
        with patch.object(file_ops, "_search_workspace", return_value=empty), \
             patch.object(file_ops, "_zettlab_nas_fallback", return_value=nas):
            out = file_ops.search("x")
    assert out is nas


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


# --- credential never leaves loopback (shared hardened transport) ------------

@pytest.mark.parametrize("bad_url", [
    "https://127.0.0.1:9090/api/v1/internal/chat/append",  # face is plain http
    "http://192.168.1.10:9090/api/v1/internal/chat/append",  # not loopback
    "http://127.attacker.example/api/v1/internal/chat/append",  # prefix trick
])
def test_non_loopback_callback_url_refused_without_request(monkeypatch, file_ops, bad_url):
    """The NAS request carries the action token: a repointed callback URL must
    yield no derived endpoint and no request at all (graceful None — the
    workspace result stands, same as any other fallback unavailability).
    NOTE: a raising sentinel would be swallowed by the fallback's blanket
    except and still return None — capture calls and assert none happened."""
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", bad_url)
    assert ShellFileOperations._zettlab_agent_search_url() is None
    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}], "total_count": 1}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
        assert file_ops._zettlab_nas_fallback("q", 50) is None
    assert "req" not in captured, f"token was sent off loopback: {captured}"


def test_nas_fallback_uses_shared_hardened_transport():
    """The NAS request must go through the shared no-proxy/no-redirect opener
    (tools.loopback_transport), not the proxy-honouring global urlopen — one
    shared primitive so the constraint cannot drift per call site."""
    import tools.file_operations as fo
    from tools.loopback_transport import urlopen_hardened as shared

    assert fo.urlopen_hardened is shared


# --- nas_search first-class entry (target='nas') ------------------------------

def test_nas_search_semantic_sends_semantic_mode_and_long_timeout(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    captured = {}
    payload = {"data": {"items": [{"path": "/nas/bird.jpg"}], "total_count": 1}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
        result = file_ops.nas_search("鸟", limit=60, semantic=True)

    assert result.total_count == 1
    sent = json.loads(captured["req"].data.decode("utf-8"))
    assert sent["modes"] == ["name", "content", "semantic"]
    # Semantic leg embeds on the device c-engine (~30s cold start): the
    # fallback's 10s timeout would turn every first photo query into a miss.
    assert captured["timeout"] == 45


def test_nas_search_default_keeps_fast_modes(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    captured = {}
    payload = {"data": {"items": [{"path": "/nas/a.pdf"}], "total_count": 1}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
        file_ops.nas_search("report", limit=60)

    sent = json.loads(captured["req"].data.decode("utf-8"))
    assert sent["modes"] == ["name", "content"]
    assert captured["timeout"] == 10


def test_nas_search_unavailable_without_token(monkeypatch, file_ops):
    """Unlike the fallback's silent None, the first-class entry must tell the
    model WHY there are no results, without ever sending a request."""
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", _APPEND_URL)
    with patch("tools.file_operations.urlopen_hardened", side_effect=AssertionError("should not call")):
        result = file_ops.nas_search("q")
    assert result.total_count == 0
    assert result.error and "unavailable" in result.error


def test_nas_search_empty_query_is_error(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    result = file_ops.nas_search("   ")
    assert result.error and "Empty" in result.error


def test_nas_search_zero_hits_returns_note_not_none(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen({"data": {"items": []}})):
        result = file_ops.nas_search("nothing", semantic=True)
    assert isinstance(result, SearchResult)
    assert result.total_count == 0
    assert result.note and "matched" in result.note.lower()


def test_nas_search_success_keeps_cards_note(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    payload = {"data": {"items": [{"path": "/nas/a.jpg"}, {"path": "/nas/b.jpg"}],
                        "total_count": 2}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        result = file_ops.nas_search("photos", semantic=True)
    assert result.total_count == 2
    assert result.note and "do not list" in result.note.lower()
    assert result.files == []


def test_nas_search_path_prefix_passthrough_and_omitted_when_empty(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    payload = {"data": {"items": [{"path": "/nas/v/a.mov"}], "total_count": 1}}

    captured = {}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
        file_ops.nas_search("mov", path_prefix="/volume1/subvol/data/Videos/quanzhou")
    sent = json.loads(captured["req"].data.decode("utf-8"))
    assert sent["path_prefix"] == "/volume1/subvol/data/Videos/quanzhou"

    captured = {}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload, captured)):
        file_ops.nas_search("mov")
    # 旧 server 字节级兼容：不传就不出现该字段。
    assert "path_prefix" not in json.loads(captured["req"].data.decode("utf-8"))


def test_nas_search_carded_false_returns_paths_and_honest_note(monkeypatch, file_ops):
    """server 报 carded=false = 没有任何卡片进聊天：必须回传路径清单 + 明确
    禁止宣称"已展示在上方"，否则模型会对用户谎报。"""
    _zettlab_env(monkeypatch)
    payload = {"data": {"items": [{"path": f"/nas/{i}.jpg"} for i in range(25)],
                        "total_count": 25, "carded": False}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        result = file_ops.nas_search("鸟", semantic=True)
    assert result.total_count == 25
    assert len(result.files) == 20  # capped
    assert result.note and "do not claim" in result.note.lower()


def test_nas_search_carded_true_or_absent_keeps_cards_contract(monkeypatch, file_ops):
    _zettlab_env(monkeypatch)
    for extra in ({}, {"carded": True}):
        payload = {"data": {"items": [{"path": "/nas/a.jpg"}], "total_count": 1, **extra}}
        with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
            result = file_ops.nas_search("q")
        assert result.files == []
        assert "do not list" in result.note.lower()


def test_nas_search_carded_key_visibility_in_tool_output(monkeypatch, file_ops):
    """to_dict 的 carded 键 = local-server 端 collector 的让位信号：
    server 报 true/false → 键存在（collector 跳过 legacy 衍生卡，防双卡）；
    老 server 无该字段 → 键缺失（collector 保留补偿行为）。"""
    _zettlab_env(monkeypatch)
    cases = [({"carded": True}, True), ({"carded": False}, False), ({}, None)]
    for extra, want in cases:
        payload = {"data": {"items": [{"path": "/nas/a.jpg"}], "total_count": 1, **extra}}
        with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
            d = file_ops.nas_search("q").to_dict()
        if want is None:
            assert "carded" not in d
        else:
            assert d["carded"] is want


def test_implicit_fallback_uncarded_returns_count_only(monkeypatch, file_ops):
    """隐式 workspace fallback 在 carded=false 时不得携带 NAS 路径（用户只是在
    grep 工作区，不该把个人文件名灌进上下文）；显式 nas_search 才给清单。"""
    _zettlab_env(monkeypatch)
    payload = {"data": {"items": [{"path": f"/nas/{i}.jpg"} for i in range(5)],
                        "total_count": 5, "carded": False}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        result = file_ops._zettlab_nas_fallback("q", 50)
    assert result.files == []
    assert "list the matched files" not in result.note.lower()
    assert "do not claim" in result.note.lower()


def test_nas_search_prefix_ignored_by_old_server_notes_unscoped(monkeypatch, file_ops):
    """老 server（响应无 carded 键）会忽略 path_prefix：note 必须声明结果未按
    文件夹收敛，不许让模型把全库计数当成文件夹内容汇报。"""
    _zettlab_env(monkeypatch)
    payload = {"data": {"items": [{"path": "/nas/a.mov"}], "total_count": 1}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        result = file_ops.nas_search("mov", path_prefix="/volume1/subvol/data/v")
    assert "ignored path_prefix" in result.note
    # 新 server 确认 carded=true 时不加该提示（prefix 已生效）。
    payload = {"data": {"items": [{"path": "/nas/a.mov"}], "total_count": 1, "carded": True}}
    with patch("tools.file_operations.urlopen_hardened", _fake_urlopen(payload)):
        result = file_ops.nas_search("mov", path_prefix="/volume1/subvol/data/v")
    assert "ignored path_prefix" not in result.note


def test_nas_side_effect_classification_and_dispatch_barrier():
    from agent.tool_result_classification import tool_may_have_side_effect
    from agent.tool_dispatch_helpers import _extract_parallel_scope_paths

    assert tool_may_have_side_effect("search_files") is False
    assert tool_may_have_side_effect("search_files", {"target": "nas"}) is True
    assert tool_may_have_side_effect("search_files", '{"target": "nas"}') is True
    assert tool_may_have_side_effect("search_files", {"target": "content"}) is False
    # 空 scope = sequential barrier：nas 调用不得与其它读并发。
    assert _extract_parallel_scope_paths("search_files", {"target": "nas"}) == []
    assert _extract_parallel_scope_paths("search_files", {}) != []


# --- search_files target='nas' dispatch ---------------------------------------

def test_search_tool_nas_target_dispatches_to_nas_search():
    from unittest.mock import patch as _patch
    from tools.file_tools import search_tool

    fake_ops = MagicMock()
    fake_ops.nas_search.return_value = SearchResult(total_count=3, note="cards rendered")
    with _patch("tools.file_tools._get_file_ops", return_value=fake_ops):
        out = json.loads(search_tool("鸟", target="nas", limit=60,
                                     semantic=True, path_prefix="/v/d",
                                     task_id="t-nas-1"))
    fake_ops.nas_search.assert_called_once_with(pattern="鸟", limit=60,
                                                semantic=True, path_prefix="/v/d")
    assert out["total_count"] == 3
    assert out["note"] == "cards rendered"


def test_search_tool_nas_target_unavailable_env_is_tool_error():
    from unittest.mock import patch as _patch
    from tools.file_tools import search_tool

    fake_ops = object()  # no nas_search attribute (upstream/non-Zettlab env)
    with _patch("tools.file_tools._get_file_ops", return_value=fake_ops):
        out = json.loads(search_tool("q", target="nas", task_id="t-nas-2"))
    assert "error" in out and "not available" in out["error"]
