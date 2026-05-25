import json
import urllib.request
from pathlib import Path

from tools import zettlab_connector_tools
from tools.zettlab_connector_tools import (
    _call_connector_tool,
    _connector_auth_headers,
    _connector_tools_from_frontmatter,
    _json_rpc,
    _is_zettlab_connector_skill,
    _normalise_tool_schema,
    ConnectorRPCError,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[2]

MIGRATED_CONNECTOR_SKILL_DIRS = [
    "skills/zettlab/authorized-connectors",
    "skills/zettlab/connector-manager",
    "skills/zettlab/github",
    "skills/zettlab/linear",
    "skills/zettlab/notion",
]


def test_normalise_tool_schema_drops_null_required_fields():
    schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string"}},
                "required": None,
            }
        },
        "required": None,
    }

    assert _normalise_tool_schema(schema) == {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string"}},
            }
        },
    }


def test_connector_tool_metadata_detection_uses_frontmatter_shape():
    frontmatter = {
        "prerequisites": {
            "tools": [
                "linear.list_issues",
                "linear.create_issue",
                "linear.update_issue",
            ],
        },
        "metadata": {
            "zettlab": {
                "connector_skill": True,
            },
        },
    }

    assert _is_zettlab_connector_skill(frontmatter)
    assert _connector_tools_from_frontmatter(frontmatter) == [
        "linear.list_issues",
        "linear.create_issue",
        "linear.update_issue",
    ]


def test_connector_preset_skills_are_not_bundled_in_runtime_repo():
    for relative_dir in MIGRATED_CONNECTOR_SKILL_DIRS:
        assert not (RUNTIME_ROOT / relative_dir).exists()
    assert (RUNTIME_ROOT / "skills/zettlab/scheduled-task-wizard/SKILL.md").exists()


def test_runtime_skills_do_not_keep_connector_migration_markers():
    for skill_file in (RUNTIME_ROOT / "skills").rglob("SKILL.md"):
        content = skill_file.read_text(encoding="utf-8")
        assert "temporary_location: hermes-agent" not in content
        assert "migration_task: connector-v1-t13" not in content


def test_connector_tool_call_proxies_to_zettlab_runtime(monkeypatch):
    calls = []

    def fake_json_rpc(method, params=None):
        calls.append((method, params))
        return {"result": {"content": [{"type": "text", "text": "ok"}]}}

    monkeypatch.setattr(zettlab_connector_tools, "_json_rpc", fake_json_rpc)

    result = json.loads(_call_connector_tool("linear.list_issues", {"first": 3}))

    assert result == {"content": [{"type": "text", "text": "ok"}]}
    assert calls == [
        (
            "tools/call",
            {"name": "linear.list_issues", "arguments": {"first": 3}},
        )
    ]


def test_connector_tool_call_surfaces_structured_connector_errors(monkeypatch):
    def fake_json_rpc(method, params=None):
        raise ConnectorRPCError(
            "denied_by_agent_policy",
            {"code": "denied_by_agent_policy", "provider": "linear"},
        )

    monkeypatch.setattr(zettlab_connector_tools, "_json_rpc", fake_json_rpc)

    result = json.loads(_call_connector_tool("linear.list_issues", {"first": 3}))

    assert result["error"] == "denied_by_agent_policy"
    assert result["connector_error"] == {
        "code": "denied_by_agent_policy",
        "provider": "linear",
    }


def test_connector_auth_headers_prefers_action_token(monkeypatch):
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "legacy-token")

    assert _connector_auth_headers() == {
        "X-Zettlab-Agent-Action-Token": "action-token"
    }


def test_connector_auth_headers_legacy_bearer_fallback(monkeypatch):
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "legacy-token")

    assert _connector_auth_headers() == {
        "Authorization": "Bearer legacy-token"
    }


def test_connector_auth_headers_missing_token(monkeypatch):
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    monkeypatch.delenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", raising=False)

    try:
        _connector_auth_headers()
    except RuntimeError as exc:
        assert str(exc) == "zettlab_connector_auth_token_missing"
    else:
        raise AssertionError("expected missing token error")


def test_json_rpc_sends_action_token_header(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"jsonrpc":"2.0","id":"zettlab-connector-tools","result":{}}'

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1:9090/api/v1/internal/connectors/rpc")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "legacy-token")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _json_rpc("tools/list")["result"] == {}
    assert captured["headers"]["X-zettlab-agent-action-token"] == "action-token"
    assert "Authorization" not in captured["headers"]
