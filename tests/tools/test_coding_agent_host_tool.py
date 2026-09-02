import json

from tools import coding_agent_host_tool as module


def test_coding_agent_host_exposes_thread_listing(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True, "result": {"threads": []}}

    def post(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(module, "get_secret", lambda name, default="": {
        "ZETTLAB_BROWSER_ACTION_URL": "http://127.0.0.1:19090",
        "ZETTLAB_AGENT_ACTION_TOKEN": "action-token",
    }.get(name, default))
    monkeypatch.setattr(module, "zettlab_browser_session_token", lambda: "session-token")
    monkeypatch.setattr(module, "get_session_env", lambda name, default="": "zettlab:user:agent:chat-1")
    monkeypatch.setattr(module.requests, "post", post)

    result = json.loads(module.coding_agent_host(
        action="threads_list",
        provider_id="codex",
        workspace_alias="repo",
        workspace_scope="recent",
        limit=50,
    ))

    assert result["success"] is True
    assert calls["url"].endswith("/api/v1/internal/pc/action")
    assert calls["kwargs"]["json"]["action"] == "coding-agent.threads.list"
    assert calls["kwargs"]["json"]["params"] == {
        "provider_id": "codex",
        "workspace_alias": "repo",
        "workspace_scope": "recent",
        "limit": 50,
    }


def test_zet_prompt_routes_coding_session_reads_away_from_computer_use():
    from gateway.platforms.zet_agent import _ZET_CODING_AGENT_SECTION

    assert "action=threads_list" in _ZET_CODING_AGENT_SECTION
    assert "必须传 `provider_id=codex` 或 `provider_id=claude_code`" in _ZET_CODING_AGENT_SECTION
    assert "不需要 `pc_node_status`、`pc_ui`" in _ZET_CODING_AGENT_SECTION
