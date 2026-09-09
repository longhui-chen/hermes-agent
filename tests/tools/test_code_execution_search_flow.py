"""Generated search stub -> authenticated RPC -> real native tool dispatch."""

import inspect
import socket
import threading
from unittest.mock import MagicMock

import pytest

from tools.code_execution_tool import generate_hermes_tools_module, _rpc_server_loop
from tools.file_operations import SearchResult


@pytest.mark.parametrize("transport", ["uds", "file"])
def test_search_stub_payload_uses_explicit_contract(transport):
    namespace = {}
    exec(generate_hermes_tools_module(["search_files"], transport=transport), namespace)
    captured = []
    namespace["_call"] = lambda tool, args: captured.append((tool, args))
    search = namespace["search_files"]
    assert inspect.signature(search).parameters["modes"].annotation is list
    query = dict(pattern="海边", scope="nas", modes=["semantic"], media_type="media",
                 region="北京", path_prefix="/nas/photos", return_references=True, limit=7)
    search(**query)
    tool, payload = captured[0]
    assert tool == "search_files"
    assert all(payload[k] == v for k, v in query.items())
    assert not {"target", "semantic", "video_semantic"} & payload.keys()
    for name in ("pattern", "scope", "modes"):
        assert inspect.signature(search).parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize("scope", ["nas", "workspace"])
def test_generated_search_roundtrips_through_native_rpc(tmp_path, monkeypatch, scope):
    # TCP is also supported by the production stub; loopback keeps this test
    # portable while still exercising the actual socket protocol and dispatcher.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    monkeypatch.setenv("HERMES_RPC_SOCKET", f"tcp://127.0.0.1:{listener.getsockname()[1]}")
    monkeypatch.setenv("HERMES_RPC_TOKEN", "test-search-rpc")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    ops = MagicMock()
    ops.nas_search.return_value = SearchResult(
        total_count=1, returned_count=1, files=["/nas/B.jpg"], complete=False,
        status="partial", truncated=True, carded=False,
        issues=[{"source": "semantic:video", "reason": "candidate_limit"}],
    )
    ops.search.return_value = SearchResult(total_count=0)
    monkeypatch.setattr("tools.file_tools._get_file_ops", lambda *a, **kw: ops)
    # Keep model_tools.handle_function_call and registry dispatch real. Only
    # the filesystem/HTTP operations behind the native tool are substituted.
    stop = threading.Event()
    calls, count = [], [0]
    thread = threading.Thread(target=_rpc_server_loop, kwargs=dict(
        server_sock=listener, task_id=f"search-rpc-{scope}", tool_call_log=calls,
        tool_call_counter=count, max_tool_calls=2,
        allowed_tools=frozenset({"search_files"}), stop_event=stop,
        rpc_token="test-search-rpc"), daemon=True)
    namespace = {}
    exec(generate_hermes_tools_module(["search_files"]), namespace)
    thread.start()
    try:
        if scope == "nas":
            result = namespace["search_files"](
                pattern="", scope="nas", modes=[], region="北京", media_type="media",
                path_prefix="/nas", return_references=True, limit=3)
            assert not result.get("error"), result
            assert result["files"] == ["/nas/B.jpg"]
            assert result["status"] == "partial" and result["complete"] is False
            assert result["issues"][0]["reason"] == "candidate_limit"
            kwargs = ops.nas_search.call_args.kwargs
            assert kwargs["modes"] == [] and kwargs["region"] == "北京"
            assert kwargs["media_type"] == "media" and kwargs["path_prefix"] == "/nas"
            assert kwargs["return_references"] is True and kwargs["limit"] == 3
            ops.search.assert_not_called()
        else:
            result = namespace["search_files"](
                pattern="needle", scope="workspace", modes=["content"],
                path=str(tmp_path), file_glob="*.py", offset=2, context=3)
            assert not result.get("error"), result
            kwargs = ops.search.call_args.kwargs
            assert kwargs["target"] == "content" and kwargs["file_glob"] == "*.py"
            assert kwargs["offset"] == 2 and kwargs["context"] == 3
            ops.nas_search.assert_not_called()
        assert count == [1] and calls[0]["tool"] == "search_files"
    finally:
        stop.set()
        if namespace.get("_sock"):
            namespace["_sock"].close()
        listener.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_execute_code_search_description_uses_current_contract():
    from tools.code_execution_tool import build_execute_code_schema
    description = build_execute_code_schema({"search_files"})["description"]
    assert "target:" not in description
    for field in ("scope", "modes", "media_type", "region", "complete/status/issues"):
        assert field in description
