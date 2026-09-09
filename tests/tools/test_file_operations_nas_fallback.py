"""Native NAS request/response flow; workspace no longer falls back to NAS."""
import json
from unittest.mock import MagicMock, patch

import pytest

from tools.file_operations import ShellFileOperations, SearchResult, set_zettlab_turn_id
from tools.search_contract import normalize_search_query


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "test-action")
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:19090/api/v1/internal/chat/append")
    return ShellFileOperations(MagicMock())


def response_for(request, *, paths=(), complete=True, issues=None, carded=True):
    filters = json.loads(request.data)
    return {"code": 200, "data": {"filters": filters, "items": [{"path": p} for p in paths],
            "total_count": len(paths), "complete": complete, "truncated": not complete,
            "issues": issues or [], "status": ("done" if paths else "empty") if complete else "partial",
            "carded": carded}}


def serve(factory, captured):
    def open_request(request, timeout):
        captured.append(request)
        response = MagicMock()
        response.read.return_value = json.dumps(factory(request)).encode()
        response.__enter__.return_value = response
        return response
    return open_request


@pytest.mark.parametrize("media,pattern,modes", [
    ("image", "", []), ("video", "", []), ("media", "", []),
    ("image", "海边", ["semantic"]), ("media", "海边", ["semantic"]),
    ("image", "海边", ["name"]), ("", "预算", ["content"]),
])
def test_native_query_flow_preserves_explicit_filters(ops, media, pattern, modes):
    captured = []
    turn_token = set_zettlab_turn_id("test-turn")
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, paths=["/volume1/subvol/data/B.jpg"]), captured)):
        result = ops.nas_search(pattern, modes, media_type=media, region="北京", return_references=True)
    set_zettlab_turn_id("")
    assert not result.error
    assert result.complete and result.carded
    assert result.files == ["/volume1/subvol/data/B.jpg"]
    sent = json.loads(captured[0].data)
    assert sent["pattern"] == pattern and sent["modes"] == modes
    assert sent["region"] == "北京" and sent.get("media_type", "") == media
    assert captured[0].headers["X-zettlab-turn-id"] == "test-turn"
    assert captured[0].full_url == "http://127.0.0.1:19090/api/v1/file/index/agent-search"


@pytest.mark.parametrize("complete,paths,reason", [
    (True, [], None), (False, [], "candidate_limit"),
    (False, ["/nas/a.jpg"], "query_failed"),
])
def test_complete_empty_and_partial_results_are_distinct(ops, complete, paths, reason):
    issues = [{"source": "semantic:video", "reason": reason}] if reason else []
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, paths=paths, complete=complete, issues=issues), [])):
        result = ops.nas_search("海边", ["semantic"], media_type="media", region="北京")
    data = result.to_dict()
    assert data["complete"] is complete
    assert data["status"] == ("empty" if complete else "partial")
    assert data["issues"] == issues
    assert data["filters"]["region"] == "北京"
    assert not result.error


@pytest.mark.parametrize("payload", [{"code": 64000}, {}, {"code": 200, "data": {}}, None])
def test_backend_failure_never_becomes_empty_or_triggers_retry(ops, payload):
    captured = []
    with patch("tools.file_operations.urlopen_hardened", serve(lambda req: payload, captured)):
        result = ops.nas_search("", [], region="北京", media_type="image")
    assert result.error and result.complete is False and result.status == "error"
    assert len(captured) == 1


def test_reference_limit_and_card_delivery_are_honest(ops):
    paths = [f"/nas/{n}.jpg" for n in range(25)]
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, paths=paths, carded=False), [])):
        result = ops.nas_search("", [], media_type="image", return_references=True)
    assert result.carded is False and result.truncated and not result.complete
    assert result.files == paths[:20]
    assert result.total_count == 25
    assert result.issues[-1]["reason"] == "reference_limit"


@pytest.mark.parametrize("bad_path", ["relative.jpg", "/nas/../other.jpg", "/outside/a.jpg"])
def test_untrusted_reference_cannot_escape_prefix(ops, bad_path):
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, paths=[bad_path]), [])):
        result = ops.nas_search("", [], media_type="image", path_prefix="/nas", return_references=True)
    assert result.error and not result.files


def test_server_must_confirm_region(ops):
    def factory(req):
        payload = response_for(req, paths=["/nas/other.jpg"])
        payload["data"]["filters"].pop("region")
        return payload
    with patch("tools.file_operations.urlopen_hardened", serve(factory, [])):
        result = ops.nas_search("", [], region="北京", media_type="image")
    assert result.error


@pytest.mark.parametrize("url", ["https://example.com/api/chat", "http://evil.invalid/api/chat", "not-a-url"])
def test_action_credential_never_sent_to_external_host(ops, monkeypatch, url):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", url)
    with patch("tools.file_operations.urlopen_hardened") as request:
        result = ops.nas_search("budget", ["content"])
    assert result.error
    request.assert_not_called()


def test_workspace_empty_never_widens_scope(ops):
    with patch.object(ops, "_search_workspace", return_value=SearchResult()) as workspace, \
         patch.object(ops, "nas_search") as nas:
        result = ops.search("missing")
    assert result.total_count == 0
    workspace.assert_called_once()
    nas.assert_not_called()


@pytest.mark.parametrize("scope,pattern,modes,media,region", [
    (None, "x", ["name"], "", ""), ("nas", "x", [], "", ""),
    ("nas", "", ["name"], "", ""), ("nas", "x", None, "", ""),
    ("nas", "x", ["semantic"], "", ""), ("nas", "x", ["unknown"], "image", ""),
    ("workspace", "x", ["content"], "", "北京"),
])
def test_invalid_combinations_are_actionable(scope, pattern, modes, media, region):
    with pytest.raises(ValueError):
        normalize_search_query(scope, pattern, modes, media, region)


def test_native_registry_dispatch_keeps_metadata_only_query(ops):
    from tools.file_tools import _handle_search_files
    captured = []
    with patch("tools.file_tools._get_file_ops", return_value=ops), patch(
        "tools.file_operations.urlopen_hardened", serve(
            lambda req: response_for(req, paths=["/nas/B.jpg"]), captured)):
        result = json.loads(_handle_search_files({"scope": "nas", "pattern": "", "modes": [],
                                                 "region": "北京", "media_type": "image"}))
    assert not result.get("error")
    assert result["filters"]["region"] == "北京"
    assert len(captured) == 1
    assert json.loads(captured[0].data)["modes"] == []


def test_path_prefix_normalized_before_request(ops):
    captured = []
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, paths=["/nas/a.jpg"]), captured)):
        result = ops.nas_search("", [], media_type="image", path_prefix="/nas/", return_references=True)
    assert not result.error
    assert json.loads(captured[0].data)["path_prefix"] == "/nas"


def test_location_index_not_ready_has_explicit_reason(ops):
    with patch("tools.file_operations.urlopen_hardened", serve(lambda req: {"code": 60033}, [])):
        result = ops.nas_search("", [], region="北京", media_type="image")
    assert result.status == "error" and not result.complete
    assert result.issues == [{"source": "region", "reason": "index_not_ready", "code": 60033}]
    assert result.filters["region"] == "北京"


def test_all_sources_failed_is_an_error_not_partial_empty(ops):
    issues = [{"source": "semantic:" + media, "reason": "query_failed", "code": 65003}
              for media in ("image", "video")]
    with patch("tools.file_operations.urlopen_hardened", serve(
        lambda req: response_for(req, complete=False, issues=issues, carded=False), [])):
        result = ops.nas_search("海边", ["semantic"], region="北京", media_type="media")
    assert result.error and result.status == "error" and not result.complete
    assert result.issues == issues and result.filters["region"] == "北京"
