"""计划 × 任务清单合一（方案 §3）：播种机制测试。

覆盖：
- TodoStore.seed_from_plan 的骨架生成 / 上限组边界对齐
- merge=false 整表重写的结构保护合并（骨架不可摧毁）
- merge=true 更新保留 plan 关联字段
- 历史回放（write merge=False 携带 plan_id 条目）后保护重新上膛
- turn 结束宿主端 in_progress 降级校正
- format_for_injection 携带组归属
- seed_pending_plan_todos 合成消息对：与 hydration 的 GHSA 配对校验兼容
- present_plan_with_meta 的 plan_id 生成与新旧 callback 签名兼容
"""

import json
from types import SimpleNamespace

from tools.todo_tool import TodoStore, MAX_TODO_ITEMS
from tools.plan_tool import present_plan_with_meta
from agent.plan_seeding import seed_pending_plan_todos, build_seed_messages


def _groups(*sizes):
    return [
        {
            "icon": "📦",
            "label": f"phase-{gi}",
            "count": n,
            "items": [f"step {gi}-{ii}" for ii in range(n)],
        }
        for gi, n in enumerate(sizes)
    ]


# ---------------------------------------------------------------- seed_from_plan


def test_seed_from_plan_builds_skeleton_with_linkage():
    store = TodoStore()
    items = store.seed_from_plan("plan01", _groups(2, 3))

    assert [i["id"] for i in items] == [
        "plan01-1-1", "plan01-1-2",
        "plan01-2-1", "plan01-2-2", "plan01-2-3",
    ]
    assert all(i["status"] == "pending" for i in items)
    assert [i["group_index"] for i in items] == [0, 0, 1, 1, 1]
    assert all(i["plan_id"] == "plan01" for i in items)
    assert store.plan_id == "plan01"


def test_seed_from_plan_caps_on_group_boundary():
    # 6 组 × 50 条 = 300 > 256：裁剪必须整组丢弃，不把一个阶段砍成半截。
    store = TodoStore()
    items = store.seed_from_plan("plan02", _groups(50, 50, 50, 50, 50, 50))
    assert len(items) == 250  # 5 whole groups; the 6th would exceed 256
    assert {i["group_index"] for i in items} == {0, 1, 2, 3, 4}


def test_seed_from_plan_truncates_within_single_oversized_group():
    # 单组超限（防御性：正常上游 plan 裁剪不会产生，但必须播出内容）。
    store = TodoStore()
    oversized = [{
        "icon": "📦",
        "label": "big",
        "count": 300,
        "items": [f"s{i}" for i in range(300)],
    }]
    items = store.seed_from_plan("plan03", oversized)
    assert len(items) == MAX_TODO_ITEMS


# ---------------------------------------------------- structure-protected merge


def test_merge_false_rewrite_cannot_destroy_seeded_skeleton():
    store = TodoStore()
    store.seed_from_plan("plan04", _groups(2, 1))

    # 模型无视骨架整表重写：改了一条状态、丢了两条、加了一条自编任务。
    store.write([
        {"id": "plan04-1-1", "content": "step 0-0", "status": "completed"},
        {"id": "extra-1", "content": "计划外任务", "status": "in_progress"},
    ], merge=False)

    items = store.read()
    ids = [i["id"] for i in items]
    # 骨架三条全部存活（遗漏 ≠ 取消），计划外条目追加在末尾。
    assert ids == ["plan04-1-1", "plan04-1-2", "plan04-2-1", "extra-1"]
    assert items[0]["status"] == "completed"
    assert items[0]["plan_id"] == "plan04"
    assert items[0]["group_index"] == 0
    assert items[1]["status"] == "pending"
    assert "group_index" not in items[3]


def test_merge_true_preserves_plan_linkage_fields():
    store = TodoStore()
    store.seed_from_plan("plan05", _groups(1))
    store.write([
        {"id": "plan05-1-1", "content": "step 0-0", "status": "in_progress"},
    ], merge=True)
    item = store.read()[0]
    assert item["status"] == "in_progress"
    assert item["plan_id"] == "plan05"
    assert item["group_index"] == 0


def test_hydration_replay_rearms_protection():
    # 历史回放：全新 store 整表写入带 plan_id 的条目 → 保护重新上膛。
    store = TodoStore()
    store.write([
        {"id": "plan06-1-1", "content": "a", "status": "pending",
         "group_index": 0, "plan_id": "plan06"},
        {"id": "plan06-1-2", "content": "b", "status": "completed",
         "group_index": 0, "plan_id": "plan06"},
    ], merge=False)
    assert store.plan_id == "plan06"

    # 上膛后的整表重写走保护合并：骨架存活。
    store.write([{"id": "x", "content": "y", "status": "pending"}], merge=False)
    ids = [i["id"] for i in store.read()]
    assert ids == ["plan06-1-1", "plan06-1-2", "x"]


def test_plain_list_without_plan_id_keeps_replace_semantics():
    # 普通清单（无 plan 关联）不受保护语义影响：replace 还是 replace。
    store = TodoStore()
    store.write([{"id": "a", "content": "1", "status": "pending"}], merge=False)
    store.write([{"id": "b", "content": "2", "status": "pending"}], merge=False)
    assert [i["id"] for i in store.read()] == ["b"]


# ------------------------------------------------------------- turn-end 校正


def test_demote_stale_in_progress():
    store = TodoStore()
    store.seed_from_plan("plan07", _groups(2))
    store.write([
        {"id": "plan07-1-1", "content": "step 0-0", "status": "in_progress"},
    ], merge=True)
    assert store.demote_stale_in_progress() is True
    assert store.read()[0]["status"] == "pending"
    # 幂等：没有 in_progress 时返回 False。
    assert store.demote_stale_in_progress() is False


# ------------------------------------------------------------- injection 格式


def test_injection_carries_group_linkage():
    store = TodoStore()
    store.seed_from_plan("plan08", _groups(1, 1))
    text = store.format_for_injection()
    assert "[group 0]" in text
    assert "[group 1]" in text


# ------------------------------------------------- seed_pending_plan_todos


class _FakeAgent(SimpleNamespace):
    def _flush_messages_to_session_db(self, messages):
        return True


def _fake_agent_with_pending(meta):
    emitted = []
    agent = _FakeAgent(
        _pending_plan_seed=meta,
        _todo_store=TodoStore(),
        todo_emit_callback=lambda todos, summary: emitted.append((todos, summary)),
    )
    return agent, emitted


def test_seed_pending_plan_todos_appends_hydratable_pair():
    meta = {"plan_id": "plan09", "title": "整理下载目录", "groups": _groups(2)}
    agent, emitted = _fake_agent_with_pending(meta)
    messages = [
        {"role": "user", "content": "帮我计划整理下载目录"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-plan", "type": "function",
             "function": {"name": "present_plan", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "present_plan", "tool_call_id": "call-plan",
         "content": "Plan presented to user."},
    ]

    seed_pending_plan_todos(agent, messages)

    # 消费掉 pending 元数据（幂等）。
    assert agent._pending_plan_seed is None
    # store 播种完成。
    assert agent._todo_store.plan_id == "plan09"
    # 合成消息对：assistant todo tool_call + tool result。
    assert messages[-2]["role"] == "assistant"
    call = messages[-2]["tool_calls"][0]
    assert call["function"]["name"] == "todo"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == call["id"]
    payload = json.loads(messages[-1]["content"])
    assert [t["plan_id"] for t in payload["todos"]] == ["plan09", "plan09"]
    # SSE 快照已推送。
    assert len(emitted) == 1
    assert emitted[0][1]["total"] == 2

    # GHSA 配对校验兼容：hydration 的 matcher 认这对消息。
    from run_agent import AIAgent
    assert AIAgent._tool_response_matches_todo_call(messages, len(messages) - 1)


def test_seed_pending_plan_todos_noop_without_meta():
    agent, emitted = _fake_agent_with_pending(None)
    messages = []
    seed_pending_plan_todos(agent, messages)
    assert messages == []
    assert emitted == []


class _FlushFailAgent(SimpleNamespace):
    def _flush_messages_to_session_db(self, messages):
        return False


def test_seed_does_not_emit_snapshot_when_persistence_fails():
    # Fail-closed（codex P1）：合成消息对没落盘就不给 App 推快照——否则用户
    # 看到可确认的清单，下一轮却 hydrate 不出骨架。
    emitted = []
    agent = _FlushFailAgent(
        _pending_plan_seed={"plan_id": "plan10", "title": "t", "groups": _groups(1)},
        _todo_store=TodoStore(),
        todo_emit_callback=lambda todos, summary: emitted.append(todos),
    )
    seed_pending_plan_todos(agent, [])
    assert emitted == []
    assert agent._incremental_persistence_failed is True


def test_turn_end_correction_writes_canonical_pair():
    # 校正只改内存的话，下一轮从历史 hydrate 出旧的 in_progress（codex P1）：
    # 必须像播种一样写一对 canonical todo 消息并在落盘成功后才推快照。
    from agent.plan_seeding import correct_stale_in_progress_at_turn_end

    emitted = []
    agent = _FakeAgent(
        _todo_store=TodoStore(),
        todo_emit_callback=lambda todos, summary: emitted.append(todos),
    )
    agent._todo_store.seed_from_plan("plan11", _groups(2))
    agent._todo_store.write(
        [{"id": "plan11-1-1", "content": "step 0-0", "status": "in_progress"}],
        merge=True,
    )
    messages = []
    correct_stale_in_progress_at_turn_end(agent, messages)

    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "todo"
    assert messages[-1]["role"] == "tool"
    payload = json.loads(messages[-1]["content"])
    statuses = {t["id"]: t["status"] for t in payload["todos"]}
    assert statuses["plan11-1-1"] == "pending"
    assert len(emitted) == 1
    # hydration 配对校验兼容。
    from run_agent import AIAgent
    assert AIAgent._tool_response_matches_todo_call(messages, len(messages) - 1)


def test_cancel_plan_items_cancels_unfinished_seeded_items():
    store = TodoStore()
    store.seed_from_plan("plan13", _groups(2))
    store.write(
        [{"id": "plan13-1-1", "content": "step 0-0", "status": "completed"}],
        merge=True,
    )
    assert store.cancel_plan_items() is True
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses["plan13-1-1"] == "completed"  # 已完成不动
    assert statuses["plan13-1-2"] == "cancelled"
    # 幂等：没有未完成条目时返回 False。
    assert store.cancel_plan_items() is False


def _mock_ack_env(monkeypatch, status: str, turn_id: str) -> None:
    def _env(name, default=""):
        if name == "HERMES_PLAN_ACK_STATUS":
            return status
        if name == "HERMES_PLAN_ACK_TURN_ID":
            return turn_id
        return default

    monkeypatch.setattr("gateway.session_context.get_session_env", _env)


def test_plan_ack_cancellation_writes_canonical_pair(monkeypatch):
    # 取消回执落地（codex P1）：cancelled 回执把播种待办整体置 cancelled 并
    # 写 canonical 对，否则下一轮 hydration 恢复出已取消计划的待办。
    from agent import plan_seeding as ps

    _mock_ack_env(monkeypatch, "cancelled", "turn-plan-14")
    emitted = []
    agent = _FakeAgent(
        _todo_store=TodoStore(),
        todo_emit_callback=lambda todos, summary: emitted.append(todos),
    )
    agent._todo_store.seed_from_plan("plan14", _groups(2), plan_turn_id="turn-plan-14")
    messages = []
    ps.apply_plan_ack_cancellation_at_turn_end(agent, messages)

    payload = json.loads(messages[-1]["content"])
    assert all(t["status"] == "cancelled" for t in payload["todos"])
    assert len(emitted) == 1
    from run_agent import AIAgent
    assert AIAgent._tool_response_matches_todo_call(messages, len(messages) - 1)


def test_plan_ack_cancellation_noop_on_turn_mismatch(monkeypatch):
    # 目标匹配（codex P1）：从旧计划卡发来的 cancelled 回执不能误杀当前计划；
    # 旧播种数据没有 plan_turn_id 时同样 fail-safe no-op。
    from agent import plan_seeding as ps

    _mock_ack_env(monkeypatch, "cancelled", "turn-plan-OLD")
    agent = _FakeAgent(_todo_store=TodoStore(), todo_emit_callback=None)
    agent._todo_store.seed_from_plan("plan16", _groups(1), plan_turn_id="turn-plan-NEW")
    messages = []
    ps.apply_plan_ack_cancellation_at_turn_end(agent, messages)
    assert messages == []
    assert agent._todo_store.read()[0]["status"] == "pending"

    # legacy：播种无 plan_turn_id → no-op。
    agent2 = _FakeAgent(_todo_store=TodoStore(), todo_emit_callback=None)
    agent2._todo_store.seed_from_plan("plan17", _groups(1))
    ps.apply_plan_ack_cancellation_at_turn_end(agent2, messages)
    assert agent2._todo_store.read()[0]["status"] == "pending"


def test_plan_ack_cancellation_noop_without_cancelled_status(monkeypatch):
    from agent import plan_seeding as ps

    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": "confirmed" if name == "HERMES_PLAN_ACK_STATUS" else default,
    )
    agent = _FakeAgent(_todo_store=TodoStore(), todo_emit_callback=None)
    agent._todo_store.seed_from_plan("plan15", _groups(1))
    messages = []
    ps.apply_plan_ack_cancellation_at_turn_end(agent, messages)
    assert messages == []
    assert agent._todo_store.read()[0]["status"] == "pending"


def test_seed_from_plan_respects_content_budget():
    # 内容预算（codex P1）：极端大计划不把 ~150KB 重复文本塞进合成消息对。
    from tools.todo_tool import MAX_SEED_CONTENT_CHARS

    big_groups = [
        {
            "icon": "📦",
            "label": f"g{gi}",
            "count": 50,
            "items": ["x" * 500 for _ in range(50)],
        }
        for gi in range(10)
    ]
    store = TodoStore()
    items = store.seed_from_plan("plan12", big_groups)
    total_chars = sum(len(i["content"]) for i in items)
    assert total_chars <= MAX_SEED_CONTENT_CHARS
    # 组边界对齐：每组 50×500=25000 > 24000 预算 → 首组组内截断兜底。
    assert {i["group_index"] for i in items} == {0}


def test_build_seed_messages_shape():
    msgs = build_seed_messages("planXY", [{"id": "planXY-1-1"}], '{"todos": []}')
    assert msgs[0]["tool_calls"][0]["id"] == "call_planseed_planXY"
    assert msgs[1]["tool_call_id"] == "call_planseed_planXY"
    assert msgs[1]["name"] == "todo"


# ------------------------------------------------------ present_plan_with_meta


def test_present_plan_with_meta_generates_plan_id_and_passes_to_callback():
    seen = {}

    def cb(title, groups, plan_id):
        seen["title"] = title
        seen["plan_id"] = plan_id

    result, meta = present_plan_with_meta(
        "整理计划", _groups(1), callback=cb, auto_execute=False,
    )
    assert meta is not None
    assert meta["plan_id"] == seen["plan_id"]
    assert len(meta["plan_id"]) == 12
    assert meta["groups"][0]["items"] == ["step 0-0"]
    assert "seeded from this plan" in result


def test_present_plan_with_meta_supports_legacy_two_arg_callback():
    seen = []
    result, meta = present_plan_with_meta(
        "整理计划", _groups(1),
        callback=lambda title, groups: seen.append(title),
        auto_execute=False,
    )
    assert seen == ["整理计划"]
    assert meta is not None and meta["plan_id"]


def test_present_plan_with_meta_no_callback_returns_no_meta():
    result, meta = present_plan_with_meta("整理计划", _groups(1))
    assert meta is None
    assert "📋 整理计划" in result
