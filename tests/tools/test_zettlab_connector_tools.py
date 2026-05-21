import json
from pathlib import Path

from agent.skill_utils import parse_frontmatter
from tools import zettlab_connector_tools
from tools.zettlab_connector_tools import (
    _call_connector_tool,
    _connector_tools_from_frontmatter,
    _is_zettlab_connector_skill,
    _normalise_tool_schema,
    ConnectorRPCError,
)


ZETTLAB_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills" / "zettlab"

PROVIDER_SKILL_TOOLS = {
    "github": [
        "github.list_repos",
        "github.list_issues",
        "github.list_commits",
        "github.create_issue",
    ],
    "linear": [
        "linear.list_issues",
        "linear.create_issue",
        "linear.update_issue",
    ],
    "notion": [
        "notion.search",
        "notion.get_page",
        "notion.create_page",
    ],
}

DIRECT_PROVIDER_HOSTS = (
    "api.github.com",
    "api.notion.com",
    "api.linear.app",
    "graphql.linear.app",
)


def _read_skill_frontmatter(name: str):
    content = (ZETTLAB_SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)
    return frontmatter, body


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


def test_provider_connector_skills_declare_runtime_tools_and_migration_marker():
    for skill_name, expected_tools in PROVIDER_SKILL_TOOLS.items():
        frontmatter, _ = _read_skill_frontmatter(skill_name)

        assert _is_zettlab_connector_skill(frontmatter)
        assert _connector_tools_from_frontmatter(frontmatter) == expected_tools

        zettlab_meta = frontmatter["metadata"]["zettlab"]
        assert zettlab_meta["temporary_location"] == "hermes-agent"
        assert zettlab_meta["migration_target"] == "dedicated-connector-skills-repo"
        assert zettlab_meta["migration_task"] == "connector-v1-t13"


def test_authorized_connector_skill_declares_full_first_batch_toolset():
    frontmatter, _ = _read_skill_frontmatter("authorized-connectors")

    expected_tools = [
        tool_name
        for tools in PROVIDER_SKILL_TOOLS.values()
        for tool_name in tools
    ]
    assert _is_zettlab_connector_skill(frontmatter)
    assert _connector_tools_from_frontmatter(frontmatter) == expected_tools


def test_connector_skill_docs_do_not_call_provider_apis_directly():
    connector_skill_names = [
        "authorized-connectors",
        *PROVIDER_SKILL_TOOLS.keys(),
    ]

    for skill_name in connector_skill_names:
        frontmatter, body = _read_skill_frontmatter(skill_name)
        content = json.dumps(frontmatter, ensure_ascii=False) + "\n" + body
        lower_content = content.lower()

        assert "do not" in lower_content
        assert "oauth" in lower_content
        assert "agent connector policy" in lower_content
        for host in DIRECT_PROVIDER_HOSTS:
            assert host not in lower_content


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
