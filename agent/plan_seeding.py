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

    # 计划呈现 turn 的（App/LS 侧）turn id：随播种条目持久化，取消回执按它
    # 匹配目标计划（codex P1——同会话先后两份计划时，从旧卡取消不能误杀
    # 当前计划）。播种运行在呈现请求的上下文内，ContextVar 可直接取。
    plan_turn_id = ""
    try:
        from gateway.session_context import zettlab_turn_id

        plan_turn_id = zettlab_turn_id()
    except Exception:
        plan_turn_id = ""

    try:
        seeded = store.seed_from_plan(plan_id, groups, plan_turn_id=plan_turn_id)
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
    # Fail-closed（codex P1）：计划卡与清单都只在落盘成功后才推给 App——否则
    # 用户看到一张只存在于内存的可确认卡片，而 flush 失败时 turn 走
    # session_persistence_failed、下一轮 hydrate 不出任何骨架。此处 flush 覆盖
    # 的 messages 同时包含 present_plan 的 tool result 与播种消息对，两者一起
    # 成为 canonical 之后再发布 UI。
    if persisted:
        emit_plan_card = meta.get("emit")
        if callable(emit_plan_card):
            try:
                emit_plan_card()
            except Exception:
                logger.exception("plan card emit failed for plan_id=%s", plan_id)
        # 播种即推送：App 在计划卡（决策点态）阶段就拿到带 plan_id 的清单数据，
        # 确认后原地演化不需要额外往返。
        _emit_todo_snapshot(agent, result_json)


def _persist_store_snapshot(
    agent: Any,
    messages: List[Dict[str, Any]],
    store: Any,
    *,
    call_prefix: str,
    stage: str,
) -> None:
    """把 store 当前状态写成一对 canonical todo 消息并 fail-closed 推送。

    只改内存 + 推 SSE 的话，下一轮新 agent 实例从历史 hydrate 出来的还是旧
    状态（codex P1）；与播种同构的消息对让 hydration / 压缩 / ACP 全链路复用。
    落盘成功才推快照。
    """
    from tools.todo_tool import todo_tool

    result_json = todo_tool(store=store)
    # 快照预算（codex P1）：这对消息不经过 maybe_persist_tool_result / turn
    # budget，store 被长计划外项塞大后快照可能超 MAX_TODO_RESULT_CHARS ——
    # 下一轮 hydration 直接跳过，刚看到的取消/校正状态回滚。超限时渐进压缩
    # store 条目内容（store 与快照保持一致，hydration 结果仍可信）。
    _SNAPSHOT_BUDGET_CHARS = 96_000
    for cap in (800, 240, 80):
        if len(result_json) <= _SNAPSHOT_BUDGET_CHARS:
            break
        if not store.compact_contents(cap):
            break
        result_json = todo_tool(store=store)
    call_id = f"{call_prefix}_{store.plan_id}_{len(messages)}"
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

    persisted = _flush_session_db_after_tool_progress(agent, messages, stage=stage)
    if persisted:
        _emit_todo_snapshot(agent, result_json)


def apply_plan_ack_cancellation_at_turn_end(
    agent: Any,
    messages: List[Dict[str, Any]],
) -> None:
    """取消回执落地（codex P1）：用户对计划卡回执 cancelled 时清掉播种待办。

    此前取消回执只撤销执行 token，播种的 pending 条目留在历史里——下一轮
    hydration 会把已取消计划恢复成待执行，合一卡与模型上下文都继续把它当
    待办。这里把该计划的未完成条目整体置 cancelled 并写 canonical 对持久化。
    """
    store = getattr(agent, "_todo_store", None)
    if store is None or not getattr(store, "plan_id", None):
        return
    try:
        from gateway.session_context import get_session_env

        status = str(get_session_env("HERMES_PLAN_ACK_STATUS") or "").strip().lower()
        ack_turn_id = str(get_session_env("HERMES_PLAN_ACK_TURN_ID") or "").strip()
    except Exception:
        return
    if status != "cancelled":
        return
    # 目标匹配（codex P1）：ack 携带的是计划呈现 turn 的 id，必须与播种时
    # 记录的 plan_turn_id 一致才取消——同会话先后两份计划时，从旧卡取消
    # 不能误杀当前计划。旧播种数据没有 plan_turn_id（滚动窗口）或 ack 缺
    # turn_id 时 fail-safe no-op（保持修复前行为：宁可不取消）。
    store_turn_id = next(
        (
            str(item.get("plan_turn_id") or "").strip()
            for item in store.read()
            if item.get("plan_turn_id")
        ),
        "",
    )
    if not ack_turn_id or not store_turn_id or ack_turn_id != store_turn_id:
        return
    try:
        if not store.cancel_plan_items():
            return
        _persist_store_snapshot(
            agent,
            messages,
            store,
            call_prefix="call_todocancel",
            stage=f"plan ack cancellation {store.plan_id}",
        )
    except Exception:
        logger.exception("plan ack cancellation failed")


def correct_stale_in_progress_at_turn_end(
    agent: Any,
    messages: List[Dict[str, Any]],
) -> None:
    """Turn 结束宿主端状态校正（Codex #21327 教训，方案 §3.5）.

    turn 结束后不该有条目还挂着「进行中」——模型忘了收尾时宿主端把残留的
    in_progress 降回 pending（宁可显示未完成，不虚报完成）。仅对计划播种清单
    生效，避免改变普通清单语义。取消回执优先处理（见
    apply_plan_ack_cancellation_at_turn_end，调用方先取消后校正）。
    """
    store = getattr(agent, "_todo_store", None)
    if store is None or not getattr(store, "plan_id", None):
        return
    try:
        if not store.demote_stale_in_progress():
            return
        _persist_store_snapshot(
            agent,
            messages,
            store,
            call_prefix="call_todofix",
            stage=f"turn-end todo correction {store.plan_id}",
        )
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
