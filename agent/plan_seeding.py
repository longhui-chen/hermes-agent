#!/usr/bin/env python3
"""Plan → Todo seeding (方案「计划 × 任务清单合一」§3 的运行时核心).

present_plan 成功呈现计划卡后，由这里用同一份计划骨架直接播种 TodoStore ——
每个计划子项生成一条带 group_index / plan_id 的 todo。层级映射由代码保证，
不再依赖「提示词求 agent 自建清单」（FND-004 详略错位的根源）。

跨 turn 存活：App 路径每个请求新建 AIAgent，TodoStore 唯一的跨 turn 恢复途径
是 run_agent._hydrate_todo_store，而它只认「与 assistant `todo` tool_call 配对
的 tool result」（GHSA-5g4g-6jrg-mw3g 的安全约束，不可绕过）。所以播种时同步
合成一对标准的 assistant todo tool_call + tool result 写进 messages —— 恢复、
压缩注入、ACP / TUI 消费全部自动复用现有链路，不触碰安全规则。manual 确认卡
模式下 turn 在计划呈现后立即结束，没有这对消息，播种会在用户点「开始执行」
的那一刻蒸发。

挂载点：conversation_loop 工具批次收尾后（见 seed_pending_plan_todos 调用处）。
不能在 tool_executor 的 present_plan 分支内就地 append —— 会插进同一批次其余
tool result 中间，破坏 assistant(tool_calls) ↔ tool 配对与批次预算统计
（messages[-num_tools:]）。
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_seed_messages(
    plan_id: str,
    seeded_items: List[Dict[str, Any]],
    result_json: str,
) -> List[Dict[str, Any]]:
    """构造合成的 assistant todo tool_call + tool result 消息对。

    对话历史里这对消息与真实 todo 调用同构：hydration 的配对校验
    （_tool_response_matches_todo_call 回扫最近一条 assistant）天然通过。
    """
    call_id = f"call_planseed_{plan_id}"
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "todo",
                    "arguments": json.dumps(
                        {"todos": seeded_items, "merge": False},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }
    from agent.tool_dispatch_helpers import make_tool_result_message

    tool_msg = make_tool_result_message("todo", result_json, call_id)
    return [assistant_msg, tool_msg]


def seed_pending_plan_todos(agent: Any, messages: List[Dict[str, Any]]) -> None:
    """Consume ``agent._pending_plan_seed`` and seed the TodoStore.

    幂等：无 pending 元数据时是 no-op。由 conversation_loop 在工具批次收尾后
    调用（present_plan 手动模式随后 break 结束 turn，auto 模式继续本 turn），
    两条路径共用这一个播种点。
    """
    meta = getattr(agent, "_pending_plan_seed", None)
    if not meta:
        return
    agent._pending_plan_seed = None

    store = getattr(agent, "_todo_store", None)
    if store is None:
        return

    plan_id = meta.get("plan_id") or ""
    groups = meta.get("groups") or []
    if not plan_id or not groups:
        return

    try:
        seeded = store.seed_from_plan(plan_id, groups)
    except Exception:
        logger.exception("plan seeding failed for plan_id=%s", plan_id)
        return
    if not seeded:
        return

    # 结果 JSON 直接用真实 todo 工具的读路径生成，保证与正常 todo result 同构
    # （hydration / 压缩折叠 / ACP、TUI 嗅探全部按同一形状消费）。
    from tools.todo_tool import todo_tool

    result_json = todo_tool(store=store)
    for msg in build_seed_messages(plan_id, seeded, result_json):
        messages.append(msg)

    from agent.tool_executor import _flush_session_db_after_tool_progress

    persisted = _flush_session_db_after_tool_progress(
        agent,
        messages,
        stage=f"plan seed todo {plan_id}",
    )
    # Fail-closed（codex P1）：合成消息对没落盘就不给 App 推快照——否则用户
    # 看到可确认的清单，下一轮却 hydrate 不出骨架。flush 失败时 helper 已置
    # _incremental_persistence_failed，conversation_loop 在播种后复查该标志，
    # 走既有 session_persistence_failed 路径终止 turn。
    if persisted:
        # 播种即推送：App 在计划卡（决策点态）阶段就拿到带 plan_id 的清单数据，
        # 确认后原地演化不需要额外往返。
        _emit_todo_snapshot(agent, result_json)


def correct_stale_in_progress_at_turn_end(
    agent: Any,
    messages: List[Dict[str, Any]],
) -> None:
    """Turn 结束宿主端状态校正（Codex #21327 教训，方案 §3.5）.

    turn 结束后不该有条目还挂着「进行中」——模型忘了收尾时宿主端把残留的
    in_progress 降回 pending（宁可显示未完成，不虚报完成）。仅对计划播种清单
    生效，避免改变普通清单语义。

    校正必须与播种一样写一对 canonical todo 消息（codex P1）：只改内存 +
    推 SSE 的话，下一轮新 agent 实例从历史 hydrate 出来的还是旧的
    in_progress，校正在刷新后被静默还原。同样 fail-closed：落盘成功才推快照。
    """
    store = getattr(agent, "_todo_store", None)
    if store is None or not getattr(store, "plan_id", None):
        return
    try:
        if not store.demote_stale_in_progress():
            return
        from tools.todo_tool import todo_tool

        result_json = todo_tool(store=store)
        call_id = f"call_todofix_{store.plan_id}_{len(messages)}"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "todo",
                    "arguments": json.dumps(
                        {"todos": store.read(), "merge": False},
                        ensure_ascii=False,
                    ),
                },
            }],
        })
        from agent.tool_dispatch_helpers import make_tool_result_message

        messages.append(make_tool_result_message("todo", result_json, call_id))

        from agent.tool_executor import _flush_session_db_after_tool_progress

        persisted = _flush_session_db_after_tool_progress(
            agent,
            messages,
            stage=f"turn-end todo correction {store.plan_id}",
        )
        if persisted:
            _emit_todo_snapshot(agent, result_json)
    except Exception:
        logger.exception("turn-end todo correction failed")


def _emit_todo_snapshot(agent: Any, result_json: str) -> None:
    """Best-effort push of the current list through todo_emit_callback."""
    emit = getattr(agent, "todo_emit_callback", None)
    if not emit:
        return
    try:
        payload = json.loads(result_json)
        emit(payload.get("todos", []), payload.get("summary", {}))
    except Exception:
        logger.debug("todo emit after seeding failed", exc_info=True)
