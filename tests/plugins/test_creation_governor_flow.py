from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.plugins import PluginManager
from tests.plugins.test_creation_governor_plugin import _load_plugin


class _Context:
    def __init__(self) -> None:
        self.tools = []
        self.hooks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))


def test_plugin_flow_registers_judgment_hooks_and_a_non_creating_proposal_tool():
    plugin = _load_plugin()
    context = _Context()
    plugin.register(context)

    assert [args[0] for args, _kwargs in context.hooks] == [
        "pre_llm_call",
        "transform_llm_output",
        "post_llm_call",
        "pre_tool_call",
    ]
    assert [tool["name"] for tool in context.tools] == ["propose_creation"]
    description = context.tools[0]["schema"]["description"]
    assert "a single ordinary task is enough" in description
    assert "same permissive rule equally to all three creation types" in description
    assert "one AI-news lookup" in description
    assert "direct scheduled instruction" in description

    result = json.loads(
        context.tools[0]["handler"](
            {
                "creation_type": "scheduled_task",
                "suggested_name": "每日竞品简报",
                "reason": "价值来自每天自动执行",
                "evidence": "用户连续讨论每日竞品变化",
                "confidence": 0.95,
                "dedup_key": "daily-competitor-brief",
            },
            session_id="flow-session",
        )
    )

    assert result["status"] == "proposal_ready"
    assert result["creation_type"] == "scheduled_task"
    assert "只输出草案" in result["next_step"]
    assert "确认创建" in result["next_step"]
    assert "create" not in context.tools[0]["name"]


def test_real_plugin_manager_loads_all_governor_hooks():
    plugin_root = Path(__file__).resolve().parents[2] / "plugins"
    manager = PluginManager()
    manifests = manager._scan_directory(plugin_root, source="bundled")
    manifest = next(item for item in manifests if item.name == "creation-governor")

    manager._load_plugin(manifest)

    loaded = manager._plugins[manifest.key or manifest.name]
    assert loaded.enabled is True
    assert loaded.error is None
    assert set(loaded.hooks_registered) == {
        "pre_llm_call",
        "transform_llm_output",
        "post_llm_call",
        "pre_tool_call",
    }
    assert loaded.tools_registered == ["propose_creation"]
