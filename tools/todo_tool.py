#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides an in-memory task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations. The state
lives on the AIAgent instance (one per session) and is re-injected into
the conversation after context compression events.

Design:
- Single `todo` tool: provide `todos` param to write, omit to read
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import json
from typing import Dict, Any, List, Optional


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Bounds on persisted todo state. The todo list is a planning aid the model
# re-reads after every context-compression event (see format_for_injection),
# so unbounded item content or count defeats the compression it rides through.
# These caps keep a single oversized item (whether authored by the model or
# replayed from caller-supplied history on the API server) from inflating the
# re-injection block. Generous relative to real plans — a todo item is a short
# task description, and active lists are a handful of items, not hundreds.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Upper bound on a single todo tool-result payload accepted during history
# hydration. The gateway/API server replays caller-supplied conversation
# history to rebuild the store, so an oversized forged result is dropped
# before it is parsed and re-injected (see AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"
# 计划播种的内容总量预算（字符）。播种会合成一对 todo tool_call/result 写进
# 对话历史（跨 turn 存活），这对消息不经过 maybe_persist_tool_result /
# enforce_turn_budget 的工具结果裁剪——极端大计划（上游上限 20 组 × 50 条 ×
# 500 字符）会产生 ~150KB 重复上下文，压爆设备端小上下文模型（codex P1）。
# 预算按组边界对齐截断；24K 字符 ≈ 正常计划（几十条 × 短句）的十倍余量。
MAX_SEED_CONTENT_CHARS = 24_000
# Persisted as ordinary message content. ContextCompressor uses this stable
# header to distinguish the synthetic post-compaction row from a real user.
TODO_INJECTION_HEADER = (
    "[Your active task list was preserved across context compression]"
)


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
      - group_index (optional): 0-based index of the plan group this item was
        seeded from (plan-linked lists only; see seed_from_plan)
      - plan_id (optional): id of the present_plan card this item belongs to

    Plan-seeded lists (seed_from_plan / replayed history carrying plan_id)
    arm structural protection: a merge=false full rewrite from the model can
    update seeded items' content/status but cannot destroy the seeded
    skeleton (ids, group linkage, order) — the skeleton is the contract the
    App renders the unified plan/todo card from.
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        # zettlab-overlay(H5-B2b): 计划播种状态字段，B2b 删除; upstream: none
        # Plan-seeding state（结构保护合并的依据）：仅当清单由计划播种（或从
        # 历史回放出带 plan_id 的条目）时非空。
        self._plan_id: Optional[str] = None
        self._plan_seeded_ids: set = set()

    @property
    def plan_id(self) -> Optional[str]:
        """plan_id of the seeded plan this list is linked to, if any."""
        return self._plan_id

    # zettlab-overlay(H5-B2b): 用计划骨架播种 todo，B2b 删除; upstream: none
    def seed_from_plan(
        self,
        plan_id: str,
        groups: List[Dict[str, Any]],
        plan_turn_id: str = "",
        budget_chars: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Seed the list from a present_plan skeleton (code-guaranteed mapping).

        每个计划子项生成一条 todo：稳定 id ``{plan_id}-{组序号}-{子项序号}``、
        pending 状态、携带 group_index 与 plan_id。替代「提示词求 agent 自建
        清单」——层级映射由代码保证（方案 §3，FND-004 详略错位的根治）。

        Cap alignment: plan 上限（20 组 × 50 条 = 1000）大于 MAX_TODO_ITEMS
        （256），另有 MAX_SEED_CONTENT_CHARS 内容总量预算（合成消息对不走工具
        结果裁剪，见常量注释）。两种裁剪都按**组边界**对齐——放不下的组整组
        丢弃，不把一个阶段砍成半截；若第一组单独超限则组内截断（必须播出内容）。

        Returns the seeded list (copy).
        """
        # plan_turn_id 长度上限（codex P1）：调用方 metadata.turn_id 无空白时
        # 长度不受限，原样复制到每个条目会让 synthetic result 膨胀到 MB 级、
        # 超 MAX_TODO_RESULT_CHARS 后下一轮 hydrate 直接跳过整份清单。与
        # _validate 同界（128），超限按缺省（取消回执退回 fail-safe no-op）。
        if plan_turn_id and len(plan_turn_id) > 128:
            plan_turn_id = ""
        budget = MAX_SEED_CONTENT_CHARS if not budget_chars or budget_chars <= 0 \
            else min(MAX_SEED_CONTENT_CHARS, int(budget_chars))
        items: List[Dict[str, str]] = []
        content_chars = 0
        for gi, group in enumerate(groups or []):
            if not isinstance(group, dict):
                continue
            group_items = [
                str(i).strip() for i in (group.get("items") or []) if str(i).strip()
            ]
            if not group_items:
                continue
            group_chars = sum(len(i) for i in group_items)
            if items and (
                len(items) + len(group_items) > MAX_TODO_ITEMS
                or content_chars + group_chars > budget
            ):
                break  # 组边界对齐：这一组放不下（条数或内容预算）就整组停止
            for ii, content in enumerate(group_items):
                if len(items) >= MAX_TODO_ITEMS or content_chars >= budget:
                    break  # 首组单独超限：组内截断兜底
                capped = self._cap_content(content)
                content_chars += len(capped)
                items.append({
                    "id": f"{plan_id}-{gi + 1}-{ii + 1}",
                    "content": capped,
                    "status": "pending",
                    "group_index": gi,
                    "plan_id": plan_id,
                    # 计划呈现 turn 的 id（取消回执的匹配目标，见 _validate 注释）。
                    **({"plan_turn_id": plan_turn_id} if plan_turn_id else {}),
                })
        self._items = items
        self._plan_id = plan_id
        self._plan_seeded_ids = {item["id"] for item in items}
        return self.read()

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        # zettlab-overlay(H5-B2b): 写入路径的播种骨架保护与解除，B2b 删除; upstream: none
        # 终态计划解除保护（codex P1）：播种条目全部 completed/cancelled 后计划
        # 已经收场，后续写入都是模型在为**新任务**记录——merge=false 继续保护会
        # 把旧骨架强行保留，merge=true 则会把新待办盖上旧 plan_id 混进已结束的
        # 计划卡。两个分支写入前统一解除。
        self._disarm_if_plan_terminal()
        if not merge:
            if self._plan_id and self._plan_seeded_ids:
                # 播种保护（方案 §3.4）：schema 鼓励模型 merge=false 整表重写，
                # 但计划骨架是 App 合一卡的渲染契约——以播种骨架为准，按 id 回
                # 填模型给的状态/内容；模型新增的条目作为计划外任务追加；骨架外
                # 的旧条目按 replace 语义被新列表取代。
                self._items = self._plan_protected_replace(todos)
            else:
                # Replace mode: new list entirely
                self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
                self._rearm_plan_seeding_from_items()
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields the LLM actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = self._cap_content(str(t["content"]).strip())
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    # 播种清单的执行期新增 = 该计划的「计划外任务」：**强制**
                    # 归到当前 plan_id 并剥掉模型自称的骨架字段（codex P1 两轮
                    # 收敛）——模型带过期/幻觉 plan_id 会让 store 出现双计划，
                    # 下一轮 _rearm 因多 ID 解除保护；自封 group_index 则能伪装
                    # 成骨架成员。骨架身份只能来自播种。
                    if self._plan_id:
                        validated["plan_id"] = self._plan_id
                        validated.pop("group_index", None)
                        validated.pop("plan_turn_id", None)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
            # 压缩后重建路径（codex P1）：canonical result 被折出窗口时 store 是
            # 空的，模型按注入块用 merge=true 回填带骨架字段的条目——此时必须
            # 重新上膛，否则 store.plan_id 一直是 None，本轮取消落地 / turn-end
            # 校正全被跳过，同轮 merge=false 也失去骨架保护。
            if not self._plan_id:
                self._rearm_plan_seeding_from_items()
        # Bound total item count so a replayed/oversized list can't grow the
        # re-injection block without limit. Keep the highest-priority head
        # (list order is priority).
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    # zettlab-overlay(H5-B2b): 计划保护的解除、重上膛与受保护替换，B2b 删除; upstream: none
    def disarm_plan_protection(self) -> bool:
        """Drop plan armament explicitly (unconfirmed / mismatched-ack expiry).

        用于「用户没确认这个计划就开始别的事」——保护继续挂着会让新任务的
        清单被旧骨架劫持（codex P1）。Returns True when armament was dropped.
        """
        if not self._plan_id:
            return False
        self._plan_id = None
        self._plan_seeded_ids = set()
        return True

    def plan_untouched(self) -> bool:
        """True when the armed plan's skeleton is still entirely pending."""
        if not (self._plan_id and self._plan_seeded_ids):
            return False
        seeded = [i for i in self._items if i["id"] in self._plan_seeded_ids]
        return bool(seeded) and all(i["status"] == "pending" for i in seeded)

    def plan_turn_id(self) -> str:
        """plan_turn_id recorded on the seeded skeleton（取消/确认目标匹配用）。"""
        for item in self._items:
            value = str(item.get("plan_turn_id") or "").strip()
            if value:
                return value
        return ""

    def _disarm_if_plan_terminal(self) -> None:
        """Drop plan armament once every seeded item is completed/cancelled."""
        if not (self._plan_id and self._plan_seeded_ids):
            return
        seeded = [i for i in self._items if i["id"] in self._plan_seeded_ids]
        if seeded and all(i["status"] in {"completed", "cancelled"} for i in seeded):
            self._plan_id = None
            self._plan_seeded_ids = set()

    def _plan_protected_replace(
        self,
        todos: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Structure-protected merge for merge=false rewrites of a seeded list.

        播种骨架（id / group_index / plan_id / 原顺序）不可被模型摧毁：
        - 命中骨架 id 的条目：回填模型给的 content / status
        - 骨架条目被整表遗漏：原样保留（遗漏 ≠ 取消，取消要显式 cancelled）
        - 模型新增条目：验证后追加末尾（计划外任务）
        """
        # 记录模型是否真的给了 content（codex P1）：状态更新常只带 id+status，
        # _validate 会把缺失 content 规范成「(no description)」，无条件覆盖会把
        # 播种的真实步骤文案洗掉并随 canonical 结果持久化。
        provided_content: set = {
            str(t.get("id", "")).strip()
            for t in todos
            if isinstance(t, dict) and str(t.get("content", "") or "").strip()
        }
        incoming_by_id: Dict[str, Dict[str, str]] = {}
        extras: List[Dict[str, str]] = []
        for t in self._dedupe_by_id(todos):
            validated = self._validate(t)
            if validated["id"] in self._plan_seeded_ids:
                incoming_by_id[validated["id"]] = validated
            else:
                # 计划外任务强制归属当前计划并剥骨架字段（同 merge 分支，codex P1）。
                if self._plan_id:
                    validated["plan_id"] = self._plan_id
                    validated.pop("group_index", None)
                    validated.pop("plan_turn_id", None)
                extras.append(validated)

        rebuilt: List[Dict[str, str]] = []
        for item in self._items:
            if item["id"] not in self._plan_seeded_ids:
                continue  # 骨架外旧条目按 replace 语义由 extras 取代
            incoming = incoming_by_id.get(item["id"])
            if incoming is not None:
                item = {
                    **item,
                    # 只有模型显式给了非空 content 才覆盖，否则保留骨架文案。
                    **({"content": incoming["content"]} if item["id"] in provided_content else {}),
                    "status": incoming["status"],
                }
            rebuilt.append(item)
        rebuilt.extend(extras)
        return rebuilt

    def _rearm_plan_seeding_from_items(self) -> None:
        """Re-arm plan protection after an unprotected replace (hydration).

        历史回放走 write(merge=False) 整表重建（run_agent._hydrate_todo_store），
        彼时 store 是全新实例、保护未上膛。回放出的条目若携带唯一 plan_id，
        据此恢复播种状态，让本 turn 后续的整表重写继续受结构保护。
        """
        plan_ids = {
            item.get("plan_id")
            for item in self._items
            if item.get("plan_id")
        }
        if len(plan_ids) == 1:
            plan_id = next(iter(plan_ids))
            # 只有**真正播种**的条目（带合法 group_index）算骨架（codex P1）：
            # 执行期新增的计划外项也被盖了同一 plan_id，若一并当成受保护骨架，
            # 原计划条目全终态、计划外项还 pending 时终态解保护判定不成立 →
            # 旧计划永远解不开、后续普通 merge=false 继续被劫持。
            seeded_ids = {
                item["id"]
                for item in self._items
                if item.get("plan_id") == plan_id and item.get("group_index") is not None
            }
            # 终态计划不重新上膛：骨架条目已全部 completed/cancelled 时计划已
            # 收场，重新上膛会把刚解除的保护又装回去（codex P1）。
            terminal = seeded_ids and all(
                item["status"] in {"completed", "cancelled"}
                for item in self._items
                if item["id"] in seeded_ids
            )
            if seeded_ids and not terminal:
                self._plan_id = plan_id
                self._plan_seeded_ids = seeded_ids
            else:
                self._plan_id = None
                self._plan_seeded_ids = set()
        else:
            self._plan_id = None
            self._plan_seeded_ids = set()

    # zettlab-overlay(H5-unowned): todo 正文按上限压缩，上游 PR 候选; upstream: none
    def compact_contents(self, max_chars_per_item: int) -> bool:
        """Truncate every item's content in place（收尾快照预算用）.

        收尾 canonical 快照不走工具结果预算，store 被计划外长文塞大后快照会
        超 MAX_TODO_RESULT_CHARS——下一轮 hydration 直接跳过这条更新，用户刚
        看到的取消/校正状态回滚（codex P1）。压缩发生在 store 本体上，快照与
        store 保持一致、hydration 结果仍然可信。Returns True when changed.
        """
        changed = False
        for item in self._items:
            content = item.get("content", "")
            if len(content) > max_chars_per_item:
                keep = max(1, max_chars_per_item - len(_TRUNCATION_MARKER))
                item["content"] = content[:keep] + _TRUNCATION_MARKER
                changed = True
        return changed

    # zettlab-overlay(H5-B2b): 整表取消播种待办，B2b 删除; upstream: none
    def cancel_plan_items(self) -> bool:
        """Cancel every unfinished item of the seeded plan（取消回执处理）.

        用户对计划卡回执 cancelled 后，播种的 pending/in_progress 条目不能继续
        当待办——否则下一轮 hydration 会把已取消计划恢复成待执行（codex P1）。
        Returns True when anything changed (caller persists + re-emits).
        """
        if not self._plan_id:
            return False
        changed = False
        for item in self._items:
            if item.get("plan_id") == self._plan_id and item.get("status") in {"pending", "in_progress"}:
                item["status"] = "cancelled"
                changed = True
        return changed

    # zettlab-overlay(H5-B2b): turn 末残留 in_progress 降回 pending，B2b 删除; upstream: none
    def demote_stale_in_progress(self) -> bool:
        """Turn-end host-side correction (Codex #21327 lesson).

        turn 结束后不该有任何条目还在「进行中」——模型忘了收尾时宿主端兜底，
        把残留的 in_progress 降回 pending（宁可显示未完成，不虚报完成）。
        Returns True when anything changed (caller should re-emit the list).
        """
        changed = False
        for item in self._items:
            if item.get("status") == "in_progress":
                item["status"] = "pending"
                changed = True
        return changed

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty.
        """
        if not self._items:
            return None

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression.
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        if not active_items:
            return None

        lines = [TODO_INJECTION_HEADER]
        # 计划关联随注入块传递（codex P1）：压缩把 canonical todo result 折叠
        # 出最近窗口后，注入块是模型重建清单的唯一来源——不带 plan_id /
        # plan_turn_id 的话，后续 todo 写入会退化成普通清单，App 合一卡与取消
        # ack 的目标绑定同时丢失。明确指示模型在每个条目上回填这两个字段。
        if self._plan_id:
            plan_turn_id = next(
                (
                    str(item.get("plan_turn_id") or "").strip()
                    for item in self._items
                    if item.get("plan_turn_id")
                ),
                "",
            )
            linkage = f"plan_id: {self._plan_id}"
            if plan_turn_id:
                linkage += f" | plan_turn_id: {plan_turn_id}"
            lines.append(
                f"[{linkage}] — when updating this list, keep each item's "
                "plan_id/plan_turn_id/group_index fields exactly as seeded."
            )
        # 计划激活时注入完整骨架（codex P1）：注入块是 canonical result 被折叠
        # 后的唯一恢复通道，只带未完成项会让模型回填出「缺了已完成/已取消步骤」
        # 的清单——合一卡进度、hydration 与取消绑定一起错位。已完成项的文案压到
        # 80 字符（进度只需要 id + status，不需要全文）控制注入体积。
        rendered = self._items if self._plan_id else active_items
        for item in rendered:
            marker = markers.get(item["status"], "[?]")
            # 计划播种条目带上组归属：长任务（恰恰是最需要计划的场景）压缩一次
            # 后，注入行是模型唯一的任务记忆——丢掉归属，后续更新就会错组
            # （v2 审查 F3）。
            group_suffix = ""
            if item.get("group_index") is not None:
                group_suffix = f" [group {item['group_index']}]"
            content = item["content"]
            if item["status"] in {"completed", "cancelled"} and len(content) > 80:
                content = content[:77] + "..."
            lines.append(f"- {marker} {item['id']}. {content} ({item['status']}){group_suffix}")

        return "\n".join(lines)

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate oversized todo content to MAX_TODO_CONTENT_CHARS.

        A single huge item would otherwise inflate the post-compression
        re-injection block (format_for_injection) without bound. Keep the
        head — the actionable part of a task description — plus a marker.
        """
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status is valid.
        Returns a clean dict with {id, content, status} plus the optional
        plan-linkage fields (group_index, plan_id) when present and valid —
        剥掉它们会让计划播种的归属信息在任何一次写入后蒸发（v2 审查 F1）。
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        validated: Dict[str, str] = {"id": item_id, "content": content, "status": status}

        # zettlab-overlay(H5-B2b): 校验并保留计划归属字段，B2b 删除; upstream: none
        group_index = item.get("group_index")
        if isinstance(group_index, bool):
            group_index = None
        if isinstance(group_index, int) and 0 <= group_index <= 10_000:
            validated["group_index"] = group_index
        elif isinstance(group_index, str):
            # 长度上限 + try（codex P1）：几千位数字串 isdigit() 会放行，但
            # int() 超 Python 整数位数限制抛 ValueError——hydration 每轮都
            # 重放历史，一条坏结果会让会话持续无法恢复。不可信值一律丢弃。
            stripped = group_index.strip()
            if stripped.isdigit() and len(stripped) <= 5:
                try:
                    parsed = int(stripped)
                    if 0 <= parsed <= 10_000:
                        validated["group_index"] = parsed
                except ValueError:
                    pass

        plan_id = item.get("plan_id")
        if isinstance(plan_id, str) and plan_id.strip():
            # id 级长度上限：plan_id 是内部生成的短 hex，历史回放里超长值一律
            # 视为伪造丢弃。
            plan_id = plan_id.strip()
            if len(plan_id) <= 64:
                validated["plan_id"] = plan_id

        plan_turn_id = item.get("plan_turn_id")
        if isinstance(plan_turn_id, str) and plan_turn_id.strip():
            # 计划呈现 turn 的（App/LS 侧）turn id：取消回执按它匹配目标计划，
            # 防止从旧卡取消误杀当前计划（codex P1）。
            plan_turn_id = plan_turn_id.strip()
            if len(plan_turn_id) <= 128:
                validated["plan_turn_id"] = plan_turn_id

        return validated

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # Non-dict items get a synthetic key so _validate can handle them
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    """
    Single entry point for the todo tool. Reads or writes depending on params.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.
        store: the TodoStore instance from the AIAgent.

    Returns:
        JSON string with the full current list and summary metadata.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        # Guard: LLM sometimes sends todos as a JSON string instead of a list
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(
                f"todos must be a list, got {type(todos).__name__}"
            )
        items = store.write(todos, merge)
    else:
        items = store.read()

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# Behavioral guidance is baked into the description so it's part of the
# static tool schema (cached, never changes mid-conversation).

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status"
                        },
                        # zettlab-overlay(H5-B2b): schema 暴露计划归属字段，B2b 删除; upstream: none
                        "group_index": {
                            "type": "integer",
                            "description": (
                                "Optional. 0-based plan-group index for items "
                                "belonging to a presented plan. Preserved "
                                "automatically on seeded items — do not change it."
                            )
                        },
                        "plan_id": {
                            "type": "string",
                            "description": (
                                "Optional. Plan linkage id on seeded items. "
                                "Preserved automatically — do not change it."
                            )
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False
            }
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
