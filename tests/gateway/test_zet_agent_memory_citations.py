"""memory.citations（需求 3.1）：采集与发射的单测。

* `_push_memory_citations`：帧形态（hermes.attachment 判别式 + memory.citations
  kind + 同 turn 恒定 id 的 upsert 语义）、条目裁剪、空集不发、背压放弃；
* `invoke_tool` 的 search_memory 分支：命中条目累积到 agent 的有界容器、
  坏结果静默、去重。
"""

import json
import queue
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platforms.zet_agent import ZetAgentAdapter


def _drain(q):
    frames = []
    while True:
        try:
            frames.append(q.get_nowait())
        except queue.Empty:
            return frames


def _items(n):
    return [
        {"id": f"e{i}", "source": "MEMORY.md", "excerpt": f"条目 {i}", "extra": "x"}
        for i in range(n)
    ]


class TestPushMemoryCitations:
    def test_frame_shape_and_stable_turn_id(self):
        q = queue.Queue()
        ok = ZetAgentAdapter._push_memory_citations(q, "t-42", "sess-1", _items(2))
        assert ok is True
        frames = _drain(q)
        assert len(frames) == 1
        marker, payload = frames[0]
        assert marker == "__tool_progress__"
        assert payload["type"] == "hermes.attachment"
        attachment = payload["attachment"]
        assert attachment["id"] == "mc-t-42"
        assert attachment["kind"] == "memory.citations"
        assert attachment["v"] == 1
        assert attachment["state"] == "active"
        assert "actions" not in attachment
        items = attachment["payload"]["items"]
        assert [i["id"] for i in items] == ["e0", "e1"]
        # 只保留窄字段（extra 被丢弃）
        assert set(items[0].keys()) == {"id", "source", "excerpt"}

    def test_trims_to_max_items_and_requires_ids(self):
        q = queue.Queue()
        many = _items(20) + [{"source": "USER.md", "excerpt": "无 id 丢弃"}]
        assert ZetAgentAdapter._push_memory_citations(q, "t", "s", many)
        attachment = _drain(q)[0][1]["attachment"]
        assert len(attachment["payload"]["items"]) == ZetAgentAdapter._MEMORY_CITATION_MAX_ITEMS

    def test_empty_or_invalid_items_do_not_emit(self):
        q = queue.Queue()
        assert ZetAgentAdapter._push_memory_citations(q, "t", "s", []) is False
        assert ZetAgentAdapter._push_memory_citations(q, "t", "s", [{"no": "id"}]) is False
        assert _drain(q) == []

    def test_missing_turn_id_falls_back_to_session_digest(self):
        q = queue.Queue()
        assert ZetAgentAdapter._push_memory_citations(q, None, "sess-9", _items(1))
        attachment = _drain(q)[0][1]["attachment"]
        assert attachment["id"].startswith("mc-")
        assert len(attachment["id"]) > 3

    def test_backpressure_skips(self):
        q = queue.Queue()
        for _ in range(ZetAgentAdapter._ATTACHMENT_STREAM_BACKLOG_MAX + 1):
            q.put(("x", {}))
        assert ZetAgentAdapter._push_memory_citations(q, "t", "s", _items(1)) is False


class TestSearchMemoryCitationCollection:
    def _invoke(self, agent, result):
        from agent.agent_runtime_helpers import invoke_tool

        with patch(
            "tools.search_memory_tool.search_memory_tool",
            return_value=json.dumps(result, ensure_ascii=False),
        ):
            invoke_tool(
                agent,
                "search_memory",
                {"query": "日报"},
                effective_task_id="task-1",
            )

    def _agent(self):
        return SimpleNamespace(
            _memory_manager=None,
            session_id="sess-1",
            interrupted=False,
        )

    def test_hits_accumulate_deduped_on_agent(self):
        agent = self._agent()
        self._invoke(agent, {"items": [{"id": "a", "source": "MEMORY.md", "excerpt": "一"}]})
        self._invoke(agent, {"items": [
            {"id": "a", "source": "MEMORY.md", "excerpt": "一（重复）"},
            {"id": "b", "source": "USER.md", "excerpt": "二"},
        ]})
        sink = agent._zet_memory_citations
        assert list(sink.keys()) == ["a", "b"]
        # 首次快照保留（去重不覆盖）
        assert sink["a"]["excerpt"] == "一"

    def test_empty_or_garbage_results_leave_no_sink(self):
        agent = self._agent()
        self._invoke(agent, {"items": []})
        assert not getattr(agent, "_zet_memory_citations", None)
        from agent.agent_runtime_helpers import invoke_tool

        with patch(
            "tools.search_memory_tool.search_memory_tool",
            return_value="not-json{{",
        ):
            invoke_tool(agent, "search_memory", {"query": "x"}, effective_task_id="t")
        assert not getattr(agent, "_zet_memory_citations", None)


def test_collect_prefetch_citations_provider_granularity():
    """预取路径引用（需求 3 边界补全）：每个 provider 分块一条引用，
    id 内容哈希幂等，空块跳过，与工具命中共用同一容器。"""
    from agent.agent_runtime_helpers import collect_prefetch_citations

    class _Agent:
        pass

    agent = _Agent()
    collect_prefetch_citations(agent, [
        ("builtin", "每周五下午开产品周会，周报要在周会前发出。"),
        ("empty", "   "),
    ])
    sink = agent._zet_memory_citations
    assert len(sink) == 1
    entry = next(iter(sink.values()))
    assert entry["source"] == "builtin"
    assert entry["id"].startswith("prefetch-builtin-")
    assert "周会" in entry["excerpt"]

    # 同轮重复登记（幂等）：同内容不产生第二条
    collect_prefetch_citations(agent, [("builtin", "每周五下午开产品周会，周报要在周会前发出。")])
    assert len(agent._zet_memory_citations) == 1


def test_collect_prefetch_citations_silent_on_bad_input():
    from agent.agent_runtime_helpers import collect_prefetch_citations

    class _Agent:
        pass

    agent = _Agent()
    collect_prefetch_citations(agent, None)
    collect_prefetch_citations(agent, [("x",)])  # 坏形状 → 静默
    assert getattr(agent, "_zet_memory_citations", {}) in ({}, getattr(agent, "_zet_memory_citations", {}))
