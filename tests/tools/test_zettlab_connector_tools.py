import json
from pathlib import Path

from tools import zettlab_connector_tools
from tools.zettlab_connector_tools import (
    _call_connector_tool,
    _connector_tools_from_frontmatter,
    _registered_skill_connector_tools,
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


def test_default_custom_dispatcher_tools_are_registered_without_profile_skill():
    tools = _registered_skill_connector_tools()

    assert "custom_connector.list_tools" in tools
    assert "custom_connector.call_tool" in tools


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
