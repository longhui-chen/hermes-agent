"""
Zet Agent platform — APIServerAdapter subclass that extends the
``/v1/chat/completions`` SSE channel with Zettlab-specific events
(reasoning, approval, clarify, conversation title) without forking
the upstream platform file.

Strategy
--------
Upstream ``api_server.py`` already mixes named SSE events into the
chat-completions stream via the ``_emit`` helper:

    if isinstance(item, tuple) and item[0] == "__tool_progress__":
        await response.write(f"event: hermes.tool.progress\\ndata: {json.dumps(item[1])}\\n\\n")
    else:
        # plain string -> chat.completion.chunk delta

This lets us interleave **arbitrary structured events** on the same
``event: hermes.tool.progress`` channel by pushing
``("__tool_progress__", payload)`` tuples into the per-request
``_stream_q`` queue, with a ``payload["type"]`` discriminator selecting
the sub-event kind.

The catch: ``_stream_q`` is a local variable inside the
``_handle_chat_completions`` method on the base class, so a subclass
cannot reach it directly. We bridge the gap by **sniffing the queue
out of an existing callback's closure**: ``tool_start_callback`` and
``stream_delta_callback`` both close over ``_stream_q``, so we walk
``cb.__closure__`` looking for a ``Queue``-like object.

This is intentionally fragile (closure layout is an implementation
detail) but **fail-safe**: when the sniff fails we simply do not
attach the extended callbacks, and the platform degrades to
upstream's stock behaviour. A WARN log fires once per agent so
regressions are visible.

Why a subclass instead of patching upstream
-------------------------------------------
We sync ``hermes-agent`` from upstream regularly. Patching
``api_server.py`` directly creates merge conflicts every release.
A subclass keeps the override surface to one method (``_create_agent``)
plus a handful of optional extension methods, and inherits everything
else (CORS, auth, ResponseStore, /v1/runs, cron, health, …) for free.

Interaction model — approval & clarify
--------------------------------------
Both are blocking: the agent thread enters a callback that pushes a
prompt event, then waits on a ``threading.Event`` for the user's
answer to arrive over a separate HTTP POST.

  agent thread                    aiohttp loop                 user
  ─────────────                   ─────────────                ────
  approval triggered
  push ("__tool_progress__", …)   _emit  →   SSE  →  hermes.approval
  threading.Event.wait(timeout)   ─────────────────────────────► user
                                                                picks
                                  POST /v1/sessions/<sid>/approval/respond
                                  resolve_gateway_approval(sid, choice)
                                  threading.Event.set()
  resumes with choice   ◄──────────────────────────────────────

Because the ``threading.Event`` lives in this adapter (not in any
queue.Queue closure), it survives across stream_q lifetime and is
addressable by ``(session_id, request_id)`` from the HTTP respond
handler. ``hermes.tools.approval`` already provides the equivalent
``register_gateway_notify`` / ``resolve_gateway_approval`` plumbing
for the danger-tool flow; clarify uses ``AIAgent.clarify_callback``
directly with a ``threading.Event``-based wait we own.

Auto-title
----------
Cheapest viable: cache the first user message per session and emit a
``conversation.title`` event the first time we observe it. Re-emit
on session reset. No LLM call — pure deterministic snippet.
"""

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web  # noqa: F401  -- import for type only
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import (
    AIOHTTP_AVAILABLE,
    APIServerAdapter,
    DEFAULT_HOST,
    MAX_REQUEST_BYTES,
    _chat_finish_reason_from_result,
    _coerce_port,
    _openai_error,
)
from gateway.platforms.base import SendResult
# ZettClaw cron event hook — monkey-patches cron.scheduler at import time
# so cron triggers POST a webhook to local-server. zero hermes main-line
# changes; see zet_agent_cron.py docstring for the full rationale.
# No-op when CRON_WEBHOOK_URL env var is unset (i.e. non-ZettClaw deploys).
from gateway.platforms import zet_agent_cron as _zet_agent_cron
_zet_agent_cron.install()

logger = logging.getLogger(__name__)


def _request_value(request: Any, key: str, default: Any = None) -> Any:
    """Read aiohttp request mapping values while tolerating simple test fakes."""
    getter = getattr(request, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return request[key]
    except Exception:
        return default


# Upper bound on the per-session last-seen-model map so a long-lived process
# with many sessions cannot grow it (or its on-disk JSON) without bound
# (Engineering Hard Rule 第 1 条 内存预算). Oldest entries are evicted LRU-style;
# an evicted session simply re-records its baseline on the next open (worst
# case: one missed identity note for a session idle past the cap).
_SEEN_MODELS_CAP = 512
# Guards lazy per-adapter creation of _seen_models_lock.
_SEEN_INIT_LOCK = threading.Lock()

# Default port for the Zet Agent platform. Distinct from API_SERVER's
# 8642 so both platforms can run side-by-side during the migration.
ZET_AGENT_DEFAULT_PORT = 7900

# How long the agent thread will block waiting for a user response to
# a clarify prompt before giving up and returning an empty string. The
# approval flow uses hermes' built-in timeout (``approval.gateway_timeout``,
# default 300s) so we don't duplicate it here.
CLARIFY_RESPONSE_TIMEOUT = 300.0

# Default approval gateway timeout in seconds — must mirror the literal
# default in tools/approval.py:1110. We read this independently so the
# expires_at_ms we publish to clients matches what the agent thread will
# actually wait for. Both reads land before the wait starts, so a stable
# config read returns the same value to both call sites.
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


def _approval_timeout_seconds() -> float:
    """Return the approval gateway timeout (seconds) from runtime config.

    Mirrors the read in tools/approval.py inside the gateway wait loop.
    Falls back to ``DEFAULT_APPROVAL_TIMEOUT_SECONDS`` when the import or
    parse fails so we never publish a degenerate deadline.
    """
    try:
        from tools.approval import _get_approval_config
        return float(_get_approval_config().get(
            "gateway_timeout", DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        ))
    except Exception:
        return DEFAULT_APPROVAL_TIMEOUT_SECONDS

# Cap the auto-title at a length the APP can render in a single line
# without truncation. Beyond that, the APP can elide.
TITLE_MAX_LEN = 60

# Name of the single MCP server that carries Zettlab connector tools
# (linear.*, etc). local-server injects this server into each agent's
# profile config.yaml at spawn (see spawn.go). POST /v1/connectors/reload
# reconnects ONLY this server so a connector-policy change is picked up
# without bouncing any other MCP server the agent may have connected.
ZETTLAB_CONNECTORS_SERVER_NAME = "zettlab_connectors"


# Zettlab APP 平台的工作风格补丁。
#
# 背景：webui 那边用 hermes 默认 SOUL（含 "admit uncertainty when appropriate"
# 这种隐式 clarify nudge），所以 agent 在面对模糊指令时会主动调 clarify
# 工具确认。App 这边 agent SOUL 是从 zettlab-local-server 的 from_template.go
# 渲染的，模板 identity 字段普遍是任务定向的人物设定（"你是 X 助手……"），
# 没有 clarify 引导，agent 倾向直接执行。
#
# clarify 工具本身两边都 enabled（toolsets.py:362-374 hermes-zet-agent
# 工具包明确包含 clarify），差距纯粹在 prompt 层面。这里通过 ephemeral
# system prompt 在 zet_agent 平台层补一段最小 disambiguation 引导，让
# 结构化变更前先确认意图，但不打扰纯对话/查询流程。
#
# 不写进 SOUL.md 是为了保留 per-agent 的灵活性 —— 用户在某个 agent 的
# 身份定位里如果显式覆盖（比如"快速执行不要确认"），那条 SOUL 仍然
# 跟在这段 addendum 后面，模型会以更靠后的、更具体的指令为准。
ZETTLAB_WORKFLOW_ADDENDUM = """\
## 工作风格

执行以下结构化变更前，先用 clarify 工具向用户确认意图（把关键参数列成 2-4 个选项让用户选）：
- 创建 / 修改 / 删除定时任务
- 删除数据、清空记录、批量操作
- 发送外部消息（邮件、IM 推送）

用户已经明确指定全部关键参数（频率、时间、目标、内容）时直接执行，无需再 clarify。
信息查询、闲聊、回答问题不要 clarify。

## 计划先行（Plan-First）

面对复杂多步任务（涉及 3 个以上阶段、不可逆操作或大量数据变更）时：
1. 先调用 `present_plan` 工具，把执行计划结构化呈现给用户（分组列出每步要做什么）。
2. 调用后立即停下，等用户明确说"开始"/"确认"/"go" 等确认信号后再执行。
3. 执行阶段用 `todo` 工具逐步记录和更新进度，每完成一步立即把对应 todo 标记为 completed。

用户说"plan 模式"、"计划模式"、"先给计划"、"先别执行"或"等我确认"时，也按上述 App 计划卡片流程处理。
不要加载名为 `plan` 的 markdown skill，也不要写 `.hermes/plans`；那是 CLI/文档计划模式，不是 Zettlab App 的确认卡片。

简单的单步请求、查询、闲聊不需要 present_plan，直接执行即可。

## 用户画像语言

写入长期用户画像（memory 工具 target="user"，即 USER.md）时，必须使用简体中文。
姓名、产品名、命令、代码标识符可以保留原文，但描述用户特征、偏好、沟通风格的正文必须写成中文。
"""


def check_zet_agent_requirements() -> bool:
    """Return True iff this platform can be started in the current process."""
    return AIOHTTP_AVAILABLE


class _ClarifyEntry:
    """One pending clarify request inside a session FIFO queue.

    The agent thread enters ``wait()`` on ``event``; the HTTP respond
    handler pops the oldest entry, stores the response, and calls
    ``event.set()`` to unblock the agent. Mirrors hermes-webui's
    api/clarify.py shape so the wire protocol stays Plan-E rev4
    compliant (no request_id; per-session FIFO).
    """

    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: Optional[str] = None


class ZetAgentAdapter(APIServerAdapter):
    """APIServerAdapter subclass with Zettlab interaction events.

    Extends the chat-completions SSE channel with four typed payloads
    multiplexed onto ``event: hermes.tool.progress``:

    - ``reasoning.delta``   — model-native reasoning text
    - ``hermes.approval``   — danger-tool approval prompts
    - ``hermes.clarify``    — agent-driven clarify questions
    - ``conversation.title`` — auto-generated session titles
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        # Reset platform identity so /health and lock files report
        # ``zet_agent`` rather than ``api_server``.
        self.platform = Platform.ZET_AGENT

        # Re-resolve host/port/key from ZET_AGENT_* env vars (super()
        # already populated them from API_SERVER_* / config.extra).
        # config.extra wins over env so users can pin a port via YAML.
        extra = config.extra or {}
        if "host" not in extra:
            self._host = os.getenv("ZET_AGENT_HOST", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        if "port" not in extra:
            raw = os.getenv("ZET_AGENT_PORT", os.getenv("API_SERVER_PORT", str(ZET_AGENT_DEFAULT_PORT)))
            self._port = _coerce_port(raw, ZET_AGENT_DEFAULT_PORT)
        if "key" not in extra:
            self._api_key = os.getenv("ZET_AGENT_KEY", os.getenv("API_SERVER_KEY", ""))

        # One-shot warn flag for closure-sniff failures.
        self._sniff_warned: bool = False

        # Pending clarify prompts: session_id -> list[_ClarifyEntry] (FIFO).
        # Plan-E rev4 wire model is "single oldest pending wins on respond";
        # the queue holds entries created concurrently within one session,
        # though in practice clarify_callback blocks the agent thread so
        # only one entry exists per session at a time.
        # Approval uses hermes' own queue (tools.approval._gateway_queues),
        # we just register a notify callback and call resolve_gateway_approval
        # from the HTTP respond handler.
        self._clarify_state_lock = threading.Lock()
        self._clarify_queues: Dict[str, List[_ClarifyEntry]] = {}

        # Mirror of the most recently pushed (and not yet resolved) prompt
        # payload per session. Used by GET /v1/sessions/{sid}/pending so a
        # reconnecting client (chat.resume path) can re-render the modal
        # for any interaction the agent thread is still blocked on.
        self._pending_lock = threading.Lock()
        self._pending_clarify: Dict[str, Dict[str, Any]] = {}
        self._pending_approval: Dict[str, Dict[str, Any]] = {}

        # Active chat-completions turns keyed by X-Hermes-Session-Id, so
        # POST /v1/sessions/{sid}/interrupt can find the running agent +
        # asyncio task and stop them on demand. Upstream api_server.py only
        # tracks /v1/runs by run_id; chat-completions has no built-in
        # session-keyed stop, which is what ZET-641 needed.
        self._session_run_lock = threading.Lock()
        self._active_session_agents: Dict[str, Any] = {}
        self._active_session_tasks: Dict[str, Any] = {}

        # Per-session sticky data: title plus the set of session_ids
        # for which we've already pushed a title (avoid duplicates).
        self._session_lock = threading.Lock()
        self._session_titles: Dict[str, str] = {}
        self._titles_pushed: set[str] = set()

    # ------------------------------------------------------------------
    # _stream_q closure sniffing
    # ------------------------------------------------------------------

    @staticmethod
    def _sniff_stream_q(*candidates) -> Optional[Any]:
        """Recover the per-request ``_stream_q`` from a callback closure.

        The base class' ``_handle_chat_completions`` builds a local
        ``_stream_q: queue.Queue`` and constructs callbacks
        (``_on_delta``, ``_on_tool_start``, ``_on_tool_complete``)
        that close over it. We walk those closures looking for the
        first object that quacks like a ``queue.Queue`` (has both
        ``put`` and ``get`` callable attributes).
        """
        seen: set[int] = set()
        for cb in candidates:
            if cb is None:
                continue
            closure = getattr(cb, "__closure__", None)
            if not closure:
                continue
            for cell in closure:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if id(val) in seen:
                    continue
                seen.add(id(val))
                if callable(getattr(val, "put", None)) and callable(getattr(val, "get", None)):
                    return val
        return None

    # ------------------------------------------------------------------
    # Title helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_title(text: str) -> str:
        text = (text or "").strip().replace("\n", " ").replace("\r", " ")
        if len(text) <= TITLE_MAX_LEN:
            return text
        return text[: TITLE_MAX_LEN - 1].rstrip() + "…"

    def _maybe_update_title(self, session_id: Optional[str], user_message: str) -> Optional[str]:
        """Cache and return a new title when the session sees its first
        non-empty user message. Returns None if no update needed.

        Zettlab system markers (e.g. [ZETTLAB:BOOTSTRAP_KICKOFF],
        [ZETTLAB:SKIP_TRIGGER], [ZETTLAB:RESUME_TRIGGER]) are skipped
        so they don't pollute the conversation.title SSE event. These
        markers are local-server-issued synthetic user messages used by
        the bootstrap interview flow; see Phase 11 design doc.
        """
        if not session_id or not user_message:
            return None
        if user_message.startswith("[ZETTLAB:"):
            return None
        candidate = self._truncate_title(user_message)
        if not candidate:
            return None
        with self._session_lock:
            if session_id in self._session_titles:
                return None
            self._session_titles[session_id] = candidate
        return candidate

    def _push_title_if_new(self, stream_q: Any, session_id: Optional[str], title: Optional[str]) -> None:
        if not session_id or not title or stream_q is None:
            return
        with self._session_lock:
            if session_id in self._titles_pushed:
                return
            self._titles_pushed.add(session_id)
        try:
            # Plan-E rev4: only `title` on the wire — the WS connection
            # is already per-session so the APP doesn't need session_id
            # echoed back; local-server's translate.go drops it anyway.
            stream_q.put((
                "__tool_progress__",
                {"type": "conversation.title", "title": title},
            ))
        except Exception:
            logger.debug("[zet_agent] title push failed", exc_info=True)

    def _make_status_cb(self, stream_q: Any, previous: Any = None):
        """Forward structured AIAgent status events onto the SSE extension lane."""
        # AIAgent instances are currently created per turn. If a future change
        # reuses them, unwrap our prior wrapper instead of chaining closures that
        # still capture an old stream_q.
        if getattr(previous, "_hermes_zettlab_status_wrapper", False):
            previous = getattr(previous, "_hermes_previous_status_callback", None)

        def _status(kind: str, payload: Any = None) -> None:
            is_compaction = kind == "context.compaction" and isinstance(payload, dict)
            previous_accepts_structured = getattr(previous, "_hermes_accepts_structured_status", False)
            if previous is not None and (not is_compaction or previous_accepts_structured):
                try:
                    previous(kind, payload)
                except Exception:
                    logger.debug("[zet_agent] previous status_callback failed", exc_info=True)
            if not is_compaction:
                return
            # Goal sidecar/index 必须在压缩轮转的当下同步迁移（codex P1）：
            # goal 行此刻已被 conversation_compression 迁到新 sid（旧行标
            # cleared），只等 post-turn hook 搬 sidecar 的话，压缩后、turn
            # 结束前 gateway 挂掉会让 reconcile 沿旧 index 找到 cleared 行并
            # 删索引 —— 新 sid 下的 active goal 从此对自愈不可见。
            if str(payload.get("state") or "") == "succeeded":
                old_sid = str(payload.get("old_session_id") or "")
                new_sid = str(payload.get("new_session_id") or "")
                if old_sid and new_sid and old_sid != new_sid:
                    try:
                        self._goals().note_compaction_rotation(old_sid, new_sid)
                    except Exception:
                        logger.debug("[zet_agent] goal compaction migration failed", exc_info=True)
            event = dict(payload)
            event["type"] = "context.compaction"
            try:
                stream_q.put(("__tool_progress__", event))
            except Exception:
                logger.debug("[zet_agent] status push failed", exc_info=True)

        setattr(_status, "_hermes_accepts_structured_status", True)
        setattr(_status, "_hermes_zettlab_status_wrapper", True)
        setattr(_status, "_hermes_previous_status_callback", previous)
        return _status

    # ------------------------------------------------------------------
    # Approval — register notify callback, resolve via HTTP respond
    # ------------------------------------------------------------------

    def _make_approval_cb(self, stream_q: Any, session_id: str):
        """Return a callable suitable for ``register_gateway_notify``.

        The hermes approval module calls our ``cb(approval_data)`` from
        the agent thread (not the event loop) just before blocking on
        the per-session queue's ``threading.Event.wait()``. We push the
        prompt onto ``_stream_q`` so the SSE writer (running on the
        loop) emits it, then return immediately — the agent's wait()
        is what actually blocks the run.

        Plan-E rev4 payload: ``{command, description, pattern_key,
        pattern_keys, expires_at_ms}`` — no request_id (per-session
        FIFO; oldest pending wins). expires_at_ms is stamped here from
        the same approval config the wait loop in tools/approval.py
        reads, so the App's countdown matches the agent's actual deadline.

        We also cache the payload in ``self._pending_approval`` so a
        reconnecting client can fetch it via the GET /pending endpoint
        and re-render the modal after a ws drop.
        """
        def _notify(approval_data: Dict[str, Any]) -> None:
            # Stamp the deadline using the same config the wait loop in
            # tools/approval.py reads. The notify callback fires
            # immediately before that wait starts, so a stable config
            # read returns the same value to both sites — clients see
            # the wall-clock time the agent will actually give up at.
            expires_at_ms = int((time.time() + _approval_timeout_seconds()) * 1000)
            payload = {
                "type": "hermes.approval",
                "command": approval_data.get("command", ""),
                "description": approval_data.get("description", ""),
                "pattern_key": approval_data.get("pattern_key", ""),
                "pattern_keys": list(approval_data.get("pattern_keys", []) or []),
                "expires_at_ms": expires_at_ms,
            }
            with self._pending_lock:
                self._pending_approval[session_id] = payload
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] approval notify push failed", exc_info=True)
            # Goal projection: a blocked approval means the loop is waiting
            # on the user — surface it on the App's goal banner (HR#3: goal
            # rounds never auto-approve). No-op for non-goal sessions.
            try:
                self._goals().on_interaction_pending(session_id)
            except Exception:
                logger.debug("[zet_agent] goal waiting projection failed", exc_info=True)

        return _notify

    # ------------------------------------------------------------------
    # Clarify — per-session FIFO queue, oldest pending wins on respond
    # ------------------------------------------------------------------

    def _make_clarify_cb(self, stream_q: Any, session_id: str):
        """Return a sync ``(question, choices) -> str`` callback.

        Appends an entry to the session's FIFO queue and pushes a
        ``hermes.clarify`` event onto ``_stream_q``, then blocks on
        the entry's ``threading.Event`` until the HTTP respond handler
        pops the entry and signals it. Timeout returns "" so a stale
        clarify never hangs the turn forever.

        Plan-E rev4 wire model: no clarify_id — the respond handler
        always answers the oldest pending entry in the session.
        """
        def _ask(question: str, choices: Optional[List[str]]) -> str:
            entry = _ClarifyEntry()
            with self._clarify_state_lock:
                self._clarify_queues.setdefault(session_id, []).append(entry)

            # Stamp the deadline using the same constant the agent
            # thread waits on a few lines below. Clients see the wall-
            # clock time we will actually give up at.
            expires_at_ms = int((time.time() + CLARIFY_RESPONSE_TIMEOUT) * 1000)
            payload = {
                "type": "hermes.clarify",
                "question": question,
                "choices_offered": list(choices or []),
                "expires_at_ms": expires_at_ms,
            }
            with self._pending_lock:
                self._pending_clarify[session_id] = payload
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] clarify push failed", exc_info=True)
                self._discard_clarify_entry(session_id, entry)
                return ""
            # Goal projection: clarify blocks the turn on user input — mirror
            # the approval hook (waiting banner; no GoalManager mutation).
            try:
                self._goals().on_interaction_pending(session_id)
            except Exception:
                logger.debug("[zet_agent] goal waiting projection failed", exc_info=True)

            resolved = entry.event.wait(timeout=CLARIFY_RESPONSE_TIMEOUT)
            if not resolved:
                logger.warning(
                    "[zet_agent] clarify timeout after %ss session=%s",
                    CLARIFY_RESPONSE_TIMEOUT, session_id,
                )
                self._discard_clarify_entry(session_id, entry)
                return ""
            return entry.response or ""

        return _ask

    # ------------------------------------------------------------------
    # Todo emit — non-blocking, fires after each todo tool call
    # ------------------------------------------------------------------

    @staticmethod
    def _make_todo_emit_cb(stream_q: Any):
        """Return a sync ``(todos, summary) -> None`` callback.

        Called by tool_executor immediately after todo_tool() returns.
        Pushes a ``hermes.todo`` event onto the SSE extension lane so
        the APP can re-render the todo panel in real time.

        Non-blocking: no threading.Event, no HTTP respond path.
        """
        def _emit(todos: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
            payload = {
                "type": "hermes.todo",
                "todos": todos,
                "summary": summary,
            }
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] todo emit push failed", exc_info=True)

        return _emit

    # ------------------------------------------------------------------
    # Plan emit — non-blocking, fires when agent calls present_plan
    # ------------------------------------------------------------------

    @staticmethod
    def _make_plan_emit_cb(stream_q: Any):
        """Return a sync ``(title, groups) -> None`` callback.

        Called by tool_executor when the agent invokes present_plan.
        Pushes a ``hermes.plan`` event onto the SSE extension lane.
        present_plan() returns a stop-and-wait instruction to the agent
        immediately after, so this callback never blocks.
        """
        def _emit(title: str, groups: List[Dict[str, Any]]) -> None:
            payload = {
                "type": "hermes.plan",
                "title": title,
                "groups": groups,
            }
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] plan emit push failed", exc_info=True)

        return _emit

    def _discard_clarify_entry(self, session_id: str, entry: _ClarifyEntry) -> None:
        """Remove an unresolved entry (push failure or timeout). The
        respond handler removes via popleft on success; this path
        handles error rollback so the queue doesn't accumulate."""
        with self._clarify_state_lock:
            queue = self._clarify_queues.get(session_id)
            if queue and entry in queue:
                queue.remove(entry)
            if queue is not None and not queue:
                self._clarify_queues.pop(session_id, None)
        with self._pending_lock:
            self._pending_clarify.pop(session_id, None)

    def _register_active_session_turn(self, session_id: Optional[str], agent_ref: list, agent_task: Any) -> None:
        """Stash the in-flight chat-completions turn so the session
        interrupt endpoint can reach it. agent_ref is the mutable
        ``[None] -> [AIAgent]`` list the base handler fills in once the
        agent is constructed; we keep the list itself (not a snapshot)
        so the interrupt picks up the agent the moment it appears."""
        if not session_id:
            return
        # Stamp the registration with its profile home (contextvar scope is
        # live here — the chat request entered via /p/{profile}): the goal
        # driver's active-turn checks must not treat ANOTHER profile's
        # same-named session as this goal's in-flight turn (codex P1).
        home = ""
        try:
            from hermes_constants import get_hermes_home

            home = str(get_hermes_home())
        except Exception:
            pass
        with self._session_run_lock:
            self._active_session_agents[session_id] = agent_ref
            self._active_session_tasks[session_id] = agent_task
            if not hasattr(self, "_active_session_homes"):
                self._active_session_homes = {}
            self._active_session_homes[session_id] = home

    def _clear_active_session_turn(self, session_id: Optional[str], agent_ref: list, agent_task: Any) -> None:
        """Drop the registration ONLY if it still points at the turn we
        registered. Guards against late-clearing a fresher turn that
        the same session has already started."""
        if not session_id:
            return
        with self._session_run_lock:
            if self._active_session_agents.get(session_id) is agent_ref:
                self._active_session_agents.pop(session_id, None)
                if hasattr(self, "_active_session_homes"):
                    self._active_session_homes.pop(session_id, None)
            if self._active_session_tasks.get(session_id) is agent_task:
                self._active_session_tasks.pop(session_id, None)

    # ------------------------------------------------------------------
    # Agent factory override
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        gateway_session_key: Optional[str] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the agent for the zet_agent platform, then attach extra callbacks.

        Body mirrors ``APIServerAdapter._create_agent`` (api_server.py:740-775)
        except the toolset key and ``platform=`` argument are bound to
        ``Platform.ZET_AGENT.value`` instead of the upstream-hardcoded
        ``"api_server"`` literal. We duplicate the body rather than calling
        ``super()._create_agent(...)`` so we never have to patch upstream
        api_server.py — keeping the override surface entirely in this fork
        file and avoiding merge conflicts on every upstream sync.

        Reasoning, clarify, and approval hooks all share the same
        sniffed ``_stream_q``. If sniff fails we degrade silently to
        upstream behaviour (no extension events, but no crash).
        """
        # 在 ephemeral_system_prompt 头部接 zettlab 工作风格 addendum。
        # 上游传进来的 ephemeral 通常是 SOUL.md / IDENTITY.md 的拼接（per-agent
        # 人格），让 addendum 在前、SOUL 在后是有意的：模型在系统提示里靠后
        # 的 instruction 优先级更高，per-agent SOUL 真要 override 这条 workflow
        # 时仍能压过去。
        ephemeral_system_prompt = (
            ZETTLAB_WORKFLOW_ADDENDUM
            + ("\n\n" + ephemeral_system_prompt if ephemeral_system_prompt else "")
        )

        # ZettClaw — 让 cronjob tool 自动设 origin: 把当前 chat session_id 注入
        # contextvars，cronjob_tools._origin_from_env 会读到 platform/chat_id
        # 自动填到 cron job.origin。否则 cron 触发时 OriginStrategy 找不到 chat
        # → 走 NewSession 兜底创 phantom session, APP 看不到推送。
        if session_id:
            try:
                from gateway.session_context import set_session_vars
                # tokens 不显式 reset — contextvars 是 task-local，task 结束自动清；
                # 同 task 内多次 _create_agent 后 set 会覆盖前值，符合预期。
                set_session_vars(
                    platform="zet_agent",
                    chat_id=session_id,
                    chat_name="",  # 暂留空，APP 这边的 chat title 不通过这条路径来
                    thread_id="",
                    user_id="",
                    user_name="",
                    session_key=session_id,
                )
            except Exception as _e:
                logger.warning("[zet_agent] set_session_vars failed (cron origin won't auto-populate): %s", _e)

        from run_agent import AIAgent
        from gateway.run import (
            _resolve_runtime_agent_kwargs,
            _resolve_gateway_model,
            _load_gateway_config,
            GatewayRunner,
        )
        from hermes_cli.tools_config import _get_platform_tools

        platform_key = Platform.ZET_AGENT.value
        runtime_kwargs = _resolve_runtime_agent_kwargs()
        reasoning_config = GatewayRunner._load_reasoning_config()
        model = _resolve_gateway_model()

        # ZET-576: apply session-level model override if present.
        # _resolve_gateway_model reads config.yaml (agent default), but
        # session overrides live in gateway_runner._session_model_overrides
        # which this adapter's _create_agent bypasses. Check it here.
        gw = getattr(self, "gateway_runner", None)
        override_key = gateway_session_key or session_id
        runtime_auxiliary_task_configs = None
        runtime_supports_vision = None
        if gw is not None and override_key:
            override = getattr(gw, "_session_model_overrides", {}).get(override_key)
            if override:
                model = override.get("model", model)
                for k in ("provider", "api_key", "base_url", "api_mode"):
                    v = override.get(k)
                    if v is not None:
                        runtime_kwargs[k] = v
                context_length = override.get("context_length")
                if context_length is not None:
                    runtime_kwargs["config_context_length"] = context_length
                auxiliary = override.get("auxiliary")
                if isinstance(auxiliary, dict):
                    runtime_auxiliary_task_configs = auxiliary
                supports_vision = override.get("supports_vision")
                if isinstance(supports_vision, bool):
                    runtime_supports_vision = supports_vision
                logger.info(
                    "session-model-override applied: session=%s model=%s",
                    override_key, model,
                )

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
        fallback_model = GatewayRunner._load_fallback_model()

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform=platform_key,
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            gateway_session_key=gateway_session_key,
            request_overrides=request_overrides,
        )
        agent.runtime_auxiliary_task_configs = runtime_auxiliary_task_configs
        agent.runtime_supports_vision = runtime_supports_vision

        stream_q = self._sniff_stream_q(
            tool_start_callback,
            tool_complete_callback,
            stream_delta_callback,
        )
        if stream_q is None:
            if not self._sniff_warned:
                logger.debug(
                    "[zet_agent] _stream_q sniff returned None for session=%s; "
                    "extension events disabled (likely /v1/runs path or upstream refactor)",
                    session_id,
                )
                self._sniff_warned = True
            return agent

        # 1. Reasoning: late-bind on the agent (AIAgent reads
        # ``self.reasoning_callback`` at runtime).
        def _reasoning_cb(text: str) -> None:
            if not text:
                return
            try:
                stream_q.put(("__tool_progress__", {"type": "reasoning.delta", "text": text}))
            except Exception:
                logger.debug("[zet_agent] reasoning_cb push failed", exc_info=True)

        try:
            agent.reasoning_callback = _reasoning_cb
        except Exception:
            logger.warning(
                "[zet_agent] failed to attach reasoning_callback; degrading",
                exc_info=True,
            )

        # 2. Structured lifecycle status: late-bind so only the sniffed
        # chat-completions stream receives the App-specific extension event.
        try:
            agent.status_callback = self._make_status_cb(
                stream_q,
                getattr(agent, "status_callback", None),
            )
        except Exception:
            logger.warning("[zet_agent] failed to attach status_callback", exc_info=True)

        # 3. Clarify: late-bind. The AIAgent invokes this only if the
        # model calls the clarify tool, so the cost of always wiring
        # it is just a closure allocation.
        if session_id:
            try:
                agent.clarify_callback = self._make_clarify_cb(stream_q, session_id)
            except Exception:
                logger.warning("[zet_agent] failed to attach clarify_callback", exc_info=True)

        # 3b. Todo emit: push hermes.todo event after each todo write/read.
        # Non-blocking — tool_executor calls this after todo_tool() returns.
        try:
            agent.todo_emit_callback = self._make_todo_emit_cb(stream_q)
        except Exception:
            logger.warning("[zet_agent] failed to attach todo_emit_callback", exc_info=True)

        # 3c. Plan emit: push hermes.plan event when present_plan is called.
        # Non-blocking — tool_executor calls this, present_plan returns
        # immediately with a stop-and-wait instruction to the agent.
        try:
            agent.plan_emit_callback = self._make_plan_emit_cb(stream_q)
        except Exception:
            logger.warning("[zet_agent] failed to attach plan_emit_callback", exc_info=True)

        # 4. Approval: register a per-session notify callback.
        # We don't unregister here because chat.completions reuses the
        # same session_id across turns; unregistration happens on
        # platform disconnect (or never, for short-lived processes).
        if session_id:
            try:
                from tools.approval import register_gateway_notify
                register_gateway_notify(session_id, self._make_approval_cb(stream_q, session_id))
            except Exception:
                logger.warning("[zet_agent] failed to register approval notify", exc_info=True)

        # 5. Auto-title is emitted in _run_agent() instead — the
        # user_message arrives there as a kwarg, but at this point in
        # _create_agent it has not been threaded through yet
        # (base _run_agent passes user_message only to run_conversation).

        return agent

    async def _run_agent(
        self,
        *,
        user_message: str = "",
        conversation_history=None,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref=None,
        gateway_session_key: Optional[str] = None,
        response_mode: Optional[str] = None,
        turn_id: Optional[str] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Wrap base ``_run_agent`` to (1) push the auto-title before
        kicking off the agent thread and (2) bind the session-scoped env
        vars hermes' approval/clarify gate reads at runtime.

        ``HERMES_SESSION_KEY`` keys the per-session approval queue so
        that the notify callback we registered in ``_create_agent``
        is found when the agent calls ``check_all_command_guards``.
        ``HERMES_EXEC_ASK`` flips the approval gate from "skip" to
        "block-and-prompt" outside CLI/gateway sessions.

        Both vars are saved and restored around the call so concurrent
        chat.completions requests do not leak each other's session key.
        Process-global env is not strictly safe under concurrency, but
        webui uses the same pattern (api/streaming.py) and the
        contention window is short enough in practice.
        """
        stream_q = self._sniff_stream_q(tool_start_callback, stream_delta_callback)
        # conversation_history 守卫：_session_titles 是进程内 cache，子进程
        # 重启（lifecycle.OnUpdate / OOM / supervisor 拉起）会清零，老会话
        # 下一条 user msg 在 cache cold 时会被误判成首句重推 title，让 App
        # 把会话列表里的首句标题换成当前这条新消息。OpenAI 兼容 API 下
        # caller 每次都传完整 history，真新会话 history 必为空 —— 只在那
        # 一刻才允许触发首句去重逻辑。App 侧另有 first-write-wins 兜底。
        if stream_q is not None and not conversation_history:
            try:
                title = self._maybe_update_title(session_id, user_message)
                self._push_title_if_new(stream_q, session_id, title)
            except Exception:
                logger.debug("[zet_agent] auto-title hook failed", exc_info=True)

        # Open-time check: if this session's effective model (override, else
        # config default) differs from the persisted last-seen value, inject a
        # one-shot identity note. Covers session- and agent-level switches,
        # survives restarts; a brand-new session just records its baseline.
        try:
            if session_id:
                _eff_model = self._effective_model(session_id, gateway_session_key)
                if _eff_model:
                    with self._seen_lock():
                        _seen = self._ensure_seen_models()
                        _prev = _seen.get(session_id)
                        if _prev != _eff_model:
                            _seen[session_id] = _eff_model
                            _seen.move_to_end(session_id)
                            while len(_seen) > _SEEN_MODELS_CAP:
                                _seen.popitem(last=False)
                            self._save_seen_models()
                        elif session_id in _seen:
                            # Touch LRU position: an active session keeping the
                            # same model must not drift to the oldest end and be
                            # evicted, which would drop its baseline and miss the
                            # next switch's note.
                            _seen.move_to_end(session_id)
                    if _prev and _prev != _eff_model:
                        _note = (
                            f"[Note: the model has changed and is now {_eff_model}. "
                            f"Adjust your self-identification accordingly.]"
                        )
                        user_message = f"{_note}\n\n{user_message}"
        except Exception:
            logger.debug("[zet_agent] model-identity note hook failed", exc_info=True)

        old_session_key = os.environ.get("HERMES_SESSION_KEY")
        old_exec_ask = os.environ.get("HERMES_EXEC_ASK")
        if session_id:
            os.environ["HERMES_SESSION_KEY"] = session_id
        os.environ.setdefault("HERMES_EXEC_ASK", "1")

        try:
            result = await super()._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                agent_ref=agent_ref,
                gateway_session_key=gateway_session_key,
                response_mode=response_mode,
                turn_id=turn_id,
                request_overrides=request_overrides,
            )
            # Goal loop post-turn hook (ZET goal driver): if this session has
            # an active persistent goal, evaluate the finished turn off the
            # event loop and report the verdict (+ continuation) to
            # local-server's advance endpoint. Fire-and-forget — a hook
            # failure must never fail the turn itself.
            try:
                final_response = ""
                effective_sid = ""
                run_ok = True
                if isinstance(result, tuple) and result and isinstance(result[0], dict):
                    r0 = result[0]
                    final_response = str(r0.get("final_response") or "")
                    effective_sid = str(r0.get("session_id") or "")
                    # 硬失败轮（provider 401/限额等，failed=True 或 completed
                    # =False 带 error）不进 judge（codex P1）：final_response
                    # 是错误文本，judge 会把 "billing exhausted" 误判成 done/
                    # blocked 终结 goal，或故障期间自驱烧轮。用 SSE 同一套
                    # 分类器保证两侧闭环互补：判为 "error" 的轮，SSE 侧必然
                    # 发 __hermes_error__ + 非 stop finish → local-server 的
                    # turn watcher 走有界重踢/park；截断（length）照常评估。
                    try:
                        run_ok = _chat_finish_reason_from_result(r0) != "error"
                    except Exception:
                        run_ok = not bool(r0.get("failed"))
                if run_ok:
                    self._goals().schedule_after_turn(
                        session_id or "",
                        user_message,
                        final_response,
                        effective_session_id=effective_sid,
                    )
                else:
                    logger.info(
                        "[zet_agent] goal post-turn hook skipped for failed turn session=%s",
                        session_id,
                    )
            except Exception:
                logger.debug("[zet_agent] goal post-turn hook failed", exc_info=True)
            return result
        finally:
            if old_session_key is None:
                os.environ.pop("HERMES_SESSION_KEY", None)
            else:
                os.environ["HERMES_SESSION_KEY"] = old_session_key
            if old_exec_ask is None:
                os.environ.pop("HERMES_EXEC_ASK", None)
            else:
                os.environ["HERMES_EXEC_ASK"] = old_exec_ask

    def _effective_model(self, session_id: Optional[str], gateway_session_key: Optional[str]) -> str:
        """Return the model this session will actually use this turn: the
        session override's model if present, else the agent-level config
        default. Mirrors the model resolution in ``_create_agent`` so the
        open-time identity check compares against what the agent really runs.
        """
        gw = getattr(self, "gateway_runner", None)
        key = gateway_session_key or session_id
        if gw is not None and key:
            try:
                override = getattr(gw, "_session_model_overrides", {}).get(key)
            except Exception:
                override = None
            if override and override.get("model"):
                return override["model"]
        try:
            from gateway.run import _resolve_gateway_model
            return _resolve_gateway_model() or ""
        except Exception:
            logger.debug("[zet_agent] _resolve_gateway_model failed", exc_info=True)
            return ""

    def _seen_lock(self) -> threading.Lock:
        """Lazily create (once, race-safe) this adapter's seen-models lock."""
        lk = getattr(self, "_seen_models_lock", None)
        if lk is None:
            with _SEEN_INIT_LOCK:
                lk = getattr(self, "_seen_models_lock", None)
                if lk is None:
                    lk = threading.Lock()
                    self._seen_models_lock = lk
        return lk

    def _ensure_seen_models(self) -> "OrderedDict[str, str]":
        """Lazily load (once) the per-session last-seen model map. Call under _seen_lock()."""
        if not getattr(self, "_seen_loaded", False):
            try:
                from gateway.session_seen_models import load_seen_models
                self._seen_models = OrderedDict(load_seen_models())
            except Exception:
                self._seen_models = OrderedDict()
                logger.debug("[zet_agent] load_seen_models failed", exc_info=True)
            self._seen_loaded = True
        return self._seen_models

    def _save_seen_models(self) -> None:
        try:
            from gateway.session_seen_models import save_seen_models
            save_seen_models(getattr(self, "_seen_models", {}))
        except Exception:
            logger.debug("[zet_agent] save_seen_models failed", exc_info=True)

    @staticmethod
    def _extract_first_user_message(agent: Any) -> str:
        """Pull the most recent user message off the agent.

        ``run_conversation`` will be invoked right after we return, with
        ``user_message=...`` as a kwarg — but we don't see that here.
        Instead, walk ``agent.session_messages`` (or its kwargs cache)
        for the first user-role entry. Fall back to empty string.
        """
        for attr in ("session_messages", "_messages"):
            msgs = getattr(agent, attr, None)
            if not msgs:
                continue
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        # OpenAI multimodal: extract text parts only.
                        return " ".join(
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
        return ""

    # ------------------------------------------------------------------
    # SSE writer override — track chat-completions turn by session_id
    # ------------------------------------------------------------------

    async def _write_sse_chat_completion(
        self, request, completion_id: str, model: str, created: int,
        stream_q, agent_task, agent_ref=None, session_id: str = None,
        gateway_session_key: str = None,
    ):
        """Register the active turn under session_id for the lifetime of
        the SSE response, then delegate to the base writer. The
        ``/v1/sessions/{sid}/interrupt`` handler looks up the same map
        to stop the agent + cancel the task.

        We register the caller-provided ``agent_ref`` list (not a copy)
        so the interrupt sees the AIAgent the moment ``_run_agent``
        fills it in. ``[None]`` is normalised on the way in so the
        register helper always has a list to stash.
        """
        active_ref = agent_ref if agent_ref is not None else [None]
        self._register_active_session_turn(session_id, active_ref, agent_task)
        try:
            return await super()._write_sse_chat_completion(
                request,
                completion_id,
                model,
                created,
                stream_q,
                agent_task,
                active_ref,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
            )
        finally:
            self._clear_active_session_turn(session_id, active_ref, agent_task)

    # ------------------------------------------------------------------
    # HTTP respond handlers — wake blocked agent threads
    # ------------------------------------------------------------------

    async def _handle_approval_respond(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/approval/respond — resolve the
        oldest pending gateway approval for the session.

        Body: ``{"choice": "once"|"session"|"always"|"deny"}``.
        Plan-E rev4: no approval_id — oldest pending wins. Returns
        ``resolved`` count (0 means nothing was pending; APP can show
        a "request expired" hint).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        choice = (body.get("choice") or "").strip()
        if choice not in {"once", "session", "always", "deny"}:
            return web.json_response(
                _openai_error('choice must be one of "once","session","always","deny"'),
                status=400,
            )

        try:
            from tools.approval import resolve_gateway_approval
        except Exception as exc:
            logger.exception("[zet_agent] tools.approval import failed")
            return web.json_response(
                _openai_error(f"approval module unavailable: {exc}", err_type="server_error"),
                status=500,
            )

        resolved = resolve_gateway_approval(session_id, choice)
        with self._pending_lock:
            self._pending_approval.pop(session_id, None)
        # Goal projection: the loop is no longer blocked on the user — flip
        # the App banner back from "waiting". No-op for non-goal sessions.
        # 仅在真的解析了 approval（resolved > 0）时才清等待标记（codex P1）：
        # gateway 重启后内存 queue 已丢、App 对旧卡片的 POST 返回 resolved=0，
        # 此时清掉 sidecar 上的 interaction_pending 会让 reconcile/自动 resume
        # 把本该等确认的 goal 继续自驱（用户点的可能还是 deny）。
        if resolved:
            try:
                await asyncio.to_thread(self._goals().on_interaction_resolved, session_id)
            except Exception:
                logger.debug("[zet_agent] goal resolved projection failed", exc_info=True)
        return web.json_response({"resolved": resolved})

    async def _handle_clarify_respond(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/clarify/respond — answer the
        oldest pending clarify prompt for the session.

        Body: ``{"response": "..."}``. Plan-E rev4: no clarify_id —
        oldest pending entry in the session FIFO is resolved. 404 when
        the queue is empty so a stale APP retry surfaces explicitly
        instead of silently dropping the answer.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        response_text = str(body.get("response", "") or "")

        with self._clarify_state_lock:
            queue = self._clarify_queues.get(session_id)
            entry = queue.pop(0) if queue else None
            if queue is not None and not queue:
                self._clarify_queues.pop(session_id, None)
        if entry is None:
            return web.json_response(
                _openai_error(
                    f"No clarify pending for session {session_id}",
                    code="clarify_not_pending",
                ),
                status=404,
            )

        entry.response = response_text
        entry.event.set()
        with self._pending_lock:
            self._pending_clarify.pop(session_id, None)
        # Goal projection: mirror the approval respond hook.
        try:
            await asyncio.to_thread(self._goals().on_interaction_resolved, session_id)
        except Exception:
            logger.debug("[zet_agent] goal resolved projection failed", exc_info=True)
        return web.json_response({"resolved": 1})

    async def _handle_pending(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id}/pending — return any approval
        and clarify prompts that are still awaiting a user response.

        Used by local-server's chat.resume path so a reconnecting
        client can re-render the modal it had open before the ws drop.
        Read-only: does not consume the entries.

        Body: ``{"approval": <payload>|null, "clarify": <payload>|null}``.
        Each payload mirrors the shape of the corresponding
        hermes.tool.progress event ``data`` object (without the
        ``type`` discriminator).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info.get("session_id", "")
        with self._pending_lock:
            ap = self._pending_approval.get(session_id)
            cl = self._pending_clarify.get(session_id)
        return web.json_response({
            "approval": ap,
            "clarify": cl,
        })

    # ------------------------------------------------------------------
    # Goal loop (persistent /goal) — driver accessor + HTTP surface
    # ------------------------------------------------------------------

    def _goals(self):
        """Lazily construct this adapter's goal-loop driver (see
        gateway/platforms/zet_agent_goals.py for the architecture)."""
        drv = getattr(self, "_zet_goal_driver", None)
        if drv is None:
            from gateway.platforms.zet_agent_goals import ZetGoalDriver
            drv = ZetGoalDriver(self)
            self._zet_goal_driver = drv
        return drv

    async def _handle_session_goal(self, request: "web.Request") -> "web.Response":
        """POST/GET /v1/sessions/{session_id}/goal — create / pause / resume /
        clear / status for the session's persistent goal. Called by
        zettlab-local-server (chat.send goal trigger + App control proxy)."""
        return await self._goals().handle_goal_route(request)

    async def _handle_session_interrupt(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/interrupt — stop the active
        chat-completions turn for ``session_id``.

        Mirrors what ``_handle_stop_run`` does for the /v1/runs API,
        but keyed by ``X-Hermes-Session-Id`` so local-server's
        ``chat.cancel`` (session-scoped) has a real interrupt path
        instead of waiting for the SSE keepalive write to fail
        (up to 30 s during quiet tool periods — see ZET-641).

        Side effects, in order:
          1. ``agent.interrupt(reason)`` — flips the agent loop's
             interrupt flag and signals in-flight tools to abort. This
             is the only step that actually stops the model + tool
             work; the disconnect path eventually does the same, just
             slowly.
          2. Drop any pending clarify entries for this session (set
             empty response + signal the event) so a thread blocked
             in ``ask_user_callback`` doesn't dangle past the
             interrupt.
          3. Resolve any pending approval as ``deny`` and tear down
             the registered process / VM workers so long-running
             tools (terminal, browser) unblock.
          4. ``task.cancel()`` — cancel the asyncio task wrapper so
             ``_write_sse_chat_completion`` exits its delta loop.

        Returns 200 with ``status: "stopping"`` when we hit at least
        one of agent/task, ``"not_running"`` otherwise (caller can
        treat both as a no-op success — idempotent).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")

        # Optional reason body (HR#4: additive — legacy callers send none).
        # reason=user_cancel means the USER pressed stop: pause any active
        # goal loop BEFORE interrupting the agent, so the interrupted turn's
        # post-turn goal hook sees status=paused and never fires another
        # round — a goal must not crawl back up after an explicit stop.
        # Timeout/disconnect interruptions never POST here, so they leave
        # the loop free to continue (judge treats the cut turn as unfinished).
        interrupt_reason = ""
        try:
            if request.can_read_body:
                body = await request.json()
                if isinstance(body, dict):
                    interrupt_reason = str(body.get("reason", "") or "")
        except Exception:
            interrupt_reason = ""
        with self._session_run_lock:
            agent_ref = self._active_session_agents.get(session_id)
            task = self._active_session_tasks.get(session_id)

        if interrupt_reason == "user_cancel":
            # Mid-turn context compaction rotates the session id and migrates
            # the goal row with it (conversation_compression →
            # migrate_goal_to_session), while local-server keeps addressing
            # the pre-rotation id. Pause under BOTH ids, or a stop pressed
            # after a rotation never lands on the goal and the loop keeps
            # self-driving (codex P1).
            interrupt_sids = [session_id]
            try:
                rotated = str(getattr(agent_ref[0], "session_id", "") or "") if agent_ref else ""
                if rotated and rotated != session_id:
                    interrupt_sids.append(rotated)
            except Exception:
                pass
            for sid in interrupt_sids:
                try:
                    await asyncio.to_thread(self._goals().on_user_interrupt, sid)
                except Exception:
                    logger.debug("[zet_agent] goal pause on interrupt failed", exc_info=True)

        agent = agent_ref[0] if agent_ref else None

        if agent is not None:
            try:
                agent.interrupt("Stop requested via Zettlab")
            except Exception:
                logger.debug("[zet_agent] session interrupt: agent.interrupt failed", exc_info=True)

        self._interrupt_pending_interactions(session_id)

        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                logger.debug("[zet_agent] session interrupt: task.cancel failed", exc_info=True)

        status = "stopping" if (agent is not None or task is not None) else "not_running"
        return web.json_response({"session_id": session_id, "status": status})

    def _interrupt_pending_interactions(self, session_id: str) -> None:
        """Best-effort cleanup of agent-thread blockers for ``session_id``.

        Without this, ``agent.interrupt()`` flips the flag but the
        agent thread may still be parked inside
        ``ask_user_callback`` / approval wait / terminal tool — none
        of which check ``_interrupt_requested`` while blocked. We
        resolve each blocker with a benign value so the thread can
        wake, see the interrupt flag, and exit the loop.

        Every step is wrapped in try/except: this runs during an
        already-failed turn, and a secondary failure here would mask
        the original interrupt status returned to the caller.
        """
        # Clarify queue: drain pending entries and signal their events
        # with empty response so the ask_user callback unblocks.
        with self._clarify_state_lock:
            clarify_queue = list(self._clarify_queues.pop(session_id, []) or [])
        for entry in clarify_queue:
            try:
                entry.response = ""
                entry.event.set()
            except Exception:
                pass

        # Approval gate: tell hermes the pending approval was denied
        # so its run loop bails. tools.approval.resolve_gateway_approval
        # is the same path /v1/sessions/{sid}/approval/respond uses.
        try:
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_id, "deny")
        except Exception:
            logger.debug("[zet_agent] session interrupt: approval cleanup failed", exc_info=True)

        # Long-running tools registered with the per-session process
        # registry / terminal VM cache.
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=session_id)
        except Exception:
            logger.debug("[zet_agent] session interrupt: process registry cleanup failed", exc_info=True)
        try:
            from tools.terminal_tool import cleanup_vm
            cleanup_vm(session_id)
        except Exception:
            logger.debug("[zet_agent] session interrupt: terminal cleanup failed", exc_info=True)

        with self._pending_lock:
            self._pending_clarify.pop(session_id, None)
            self._pending_approval.pop(session_id, None)

    # ------------------------------------------------------------------
    # Diagnostic wrapper around base /v1/chat/completions
    # ------------------------------------------------------------------

    async def _diagnostic_chat_completions(self, request: "web.Request") -> "web.Response":
        """Pre-read the body so a parse / size failure surfaces the real
        exception type and a body sample in our logs, instead of bubbling
        up as the base handler's catch-all 400 "Invalid JSON in request
        body" — which masks ``RequestEntityTooLarge``, ``UnicodeDecodeError``,
        and read timeouts indistinguishably.

        ``request.read()`` caches the bytes on the Request object, so the
        downstream ``await request.json()`` inside the base handler reuses
        them — we only pay one read.
        """
        try:
            raw = await request.read()
        except Exception as e:
            cl = request.headers.get("Content-Length")
            ct = request.headers.get("Content-Type")
            logger.error(
                "[zet_agent] /v1/chat/completions body read failed: %s: %s "
                "(Content-Length=%s, Content-Type=%s, client_max_size=%s)",
                type(e).__name__, e, cl, ct, MAX_REQUEST_BYTES,
            )
            return web.json_response(
                _openai_error(
                    f"Request body could not be read ({type(e).__name__}); "
                    f"check Content-Length vs server client_max_size={MAX_REQUEST_BYTES}",
                    code="body_read_failed",
                ),
                status=413 if "TooLarge" in type(e).__name__ else 400,
            )
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            sample = raw[:256]
            try:
                sample_repr = sample.decode("utf-8", errors="replace")
            except Exception:
                sample_repr = repr(sample)
            logger.error(
                "[zet_agent] /v1/chat/completions JSON decode failed at pos %s: %s "
                "(body bytes=%d, head=%r)",
                getattr(e, "pos", "?"), e.msg, len(raw), sample_repr,
            )
            # Fall through to base handler so the client still gets the
            # original 400 shape — we just have a real log now.
        return await self._handle_chat_completions(request)

    # ------------------------------------------------------------------
    # model switch — agent-level runtime swap via local-server
    # ------------------------------------------------------------------

    async def _handle_model_switch(self, request: "web.Request") -> "web.Response":
        """POST /v1/model/switch — agent-level default model switch.

        Called by zettlab-local-server's PUT /api/v1/agent/agents/:id/model
        endpoint. Only updates the profile's config.yaml model.* slot.

        Sessions without a session-level override will pick up the new
        default on their next _create_agent call (reads config.yaml).
        Sessions WITH an override keep their override — aligning with
        hermes CLI /model --global behavior.

        Expected body: {"model": "...", "provider": "...", "base_url": "...", "api_key": "...", "api_mode"?: "...", "context_length"?: 123}
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)

        new_model = body.get("model", "")
        new_provider = body.get("provider", "")
        new_base_url = body.get("base_url", "")
        new_api_key = body.get("api_key", "")
        new_api_mode = body.get("api_mode", "")
        new_context_length = body.get("context_length", None)
        if not new_model:
            return web.json_response({"ok": False, "error": "model is required"}, status=400)

        # Update profile config.yaml so the change persists across gateway
        # restarts and new sessions / sessions without override read the
        # right default. Session-level overrides are NOT cleared — they
        # take precedence per session (same as hermes /model --global).
        try:
            from gateway.run import _load_gateway_config, _hermes_home
            from hermes_constants import get_hermes_home_override
            from utils import atomic_yaml_write
            cfg = _load_gateway_config()
            model_slot = cfg.get("model", {})
            if isinstance(model_slot, str):
                model_slot = {"default": model_slot}
            model_slot["default"] = new_model
            if new_provider:
                model_slot["provider"] = new_provider
            if new_base_url:
                model_slot["base_url"] = new_base_url
            if new_api_key:
                model_slot["api_key"] = new_api_key
            if "api_mode" in body:
                if new_api_mode:
                    model_slot["api_mode"] = new_api_mode
                else:
                    model_slot.pop("api_mode", None)
            if "context_length" in body:
                parsed_context_length = 0
                if new_context_length is not None and str(new_context_length).strip() != "":
                    parsed_context_length = int(new_context_length)
                if parsed_context_length > 0:
                    model_slot["context_length"] = parsed_context_length
                else:
                    model_slot.pop("context_length", None)
            cfg["model"] = model_slot
            override_home = get_hermes_home_override()
            config_path = (Path(override_home) if override_home else _hermes_home) / "config.yaml"
            atomic_yaml_write(config_path, cfg)
        except Exception as exc:
            logger.warning("model-switch: config write failed: %s", exc)
            return web.json_response({"ok": False, "error": f"config write: {exc}"}, status=500)

        # Identity note is handled by _run_agent's open-time compare, not here.
        logger.info(
            "model-switch: model=%s provider=%s (config.yaml only, session overrides preserved)",
            new_model, new_provider,
        )
        return web.json_response({
            "ok": True, "model": new_model,
        })

    async def _handle_session_model_switch(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/model/switch — session-level model override.

        Unlike the agent-level POST /v1/model/switch, this only changes the
        model for a single session without touching config.yaml or other
        sessions. Local-server owns persistence in
        ``session_model_overrides.json``; Hermes only applies the runtime
        override in memory so the next turn in this session picks up the
        new model.

        Expected body: {"model": "...", "provider"?: "...", "base_url"?: "...", "api_key"?: "...", "api_mode"?: "..."}
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return web.json_response(
                _openai_error("session_id is required"), status=400,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)

        new_model = body.get("model", "")
        if not new_model:
            return web.json_response({"ok": False, "error": "model is required"}, status=400)

        new_provider = body.get("provider", "")
        new_base_url = body.get("base_url", "")
        new_api_key = body.get("api_key", "")
        new_api_mode = body.get("api_mode", "")
        new_context_length = body.get("context_length", None)
        new_supports_vision = body.get("supports_vision", None)
        new_auxiliary = body.get("auxiliary", None)

        # Build the override dict — only include keys that were provided.
        override: Dict[str, Any] = {"model": new_model}
        if new_provider:
            override["provider"] = new_provider
        if new_base_url:
            override["base_url"] = new_base_url
        if new_api_key:
            override["api_key"] = new_api_key
        if new_api_mode:
            override["api_mode"] = new_api_mode
        if new_context_length is not None:
            override["context_length"] = new_context_length
        if isinstance(new_supports_vision, bool):
            override["supports_vision"] = new_supports_vision
        if isinstance(new_auxiliary, dict):
            override["auxiliary"] = new_auxiliary

        # Store override in gateway_runner so the next _create_agent call
        # for this session reads the overridden model.
        gw = getattr(self, "gateway_runner", None)
        if gw is not None:
            overrides = getattr(gw, "_session_model_overrides", None)
            if overrides is not None:
                overrides[session_id] = override
            evict = getattr(gw, "_evict_cached_agent", None)
            if callable(evict):
                try:
                    evict(session_id)
                except Exception as exc:
                    logger.warning(
                        "session-model-switch: evict_cached_agent failed for %s: %s",
                        session_id, exc,
                    )

        # Note injection happens in _run_agent (open-time compare); here we only
        # persist the override and evict the cached agent so the next turn
        # rebuilds with the new model.
        logger.info(
            "session-model-switch: session=%s model=%s provider=%s",
            session_id, new_model, new_provider,
        )
        return web.json_response({
            "ok": True,
            "session_id": session_id,
            "model": new_model,
        })

    async def _handle_session_model_clear(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/sessions/{session_id}/model — clear a session model override.

        The inverse of POST /v1/sessions/{session_id}/model/switch: drop the
        in-memory override for this single session so the next turn falls back
        to the agent default (config.yaml ``model.default``) — the exact same
        fallback a fresh session uses (see ``_resolve_turn_agent_config``,
        which re-resolves the config model every turn and only overlays an
        override when one is present).

        Conversation history is preserved — unlike ``/new`` and ``/reset``,
        which also wipe the session. We only pop the override + evict the
        cached agent so the next turn rebuilds on the default model.

        Local-server owns persistence: it deletes the entry from
        ``session_model_overrides.json`` and calls this to drop the live
        override. A session with no override is a successful no-op
        (idempotent — DELETE semantics).

        Auth: ZET_AGENT_KEY Bearer (same as the switch endpoint).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return web.json_response(
                _openai_error("session_id is required"), status=400,
            )

        cleared = False
        gw = getattr(self, "gateway_runner", None)
        if gw is not None:
            overrides = getattr(gw, "_session_model_overrides", None)
            if overrides is not None and session_id in overrides:
                overrides.pop(session_id, None)
                cleared = True
            # Only evict when we actually removed an override: an un-overridden
            # session's cached agent is already built on the default model, so
            # dropping it would force a needless rebuild.
            if cleared:
                evict = getattr(gw, "_evict_cached_agent", None)
                if callable(evict):
                    try:
                        evict(session_id)
                    except Exception as exc:
                        logger.warning(
                            "session-model-clear: evict_cached_agent failed for %s: %s",
                            session_id, exc,
                        )

        logger.info(
            "session-model-clear: session=%s cleared=%s", session_id, cleared,
        )
        return web.json_response({
            "ok": True,
            "session_id": session_id,
            "cleared": cleared,
        })

    # ------------------------------------------------------------------
    # ZET-900 — skill / connector reload control endpoints
    # ------------------------------------------------------------------

    async def _handle_skills_reload(self, request: "web.Request") -> "web.Response":
        """POST /v1/skills/reload — drop the skills prompt cache + rescan,
        and force every session (including ones currently chatting) to
        rebuild its system prompt on the next turn.

        Called by zettlab-local-server right after a skillhub install/uninstall
        lands a bundle in ``<profile>/skills/__skillhub__/...`` (see ZET-900
        plan §4.1b). The install path is pure Go and never touches this
        process, so without this nudge the in-process skills index stays stale
        until the next cold start.

        Behaviour:
          - ``clear_skills_system_prompt_cache(clear_snapshot=True)`` drops the
            in-process LRU and the on-disk snapshot, so a freshly-built system
            prompt re-scans the on-disk skill set.
          - ``scan_skill_commands()`` refreshes the slash-command table and
            gives us the current skill count for the response.
          - ``SessionDB.clear_all_system_prompts()`` nulls every session's
            stored system_prompt so the continuing-session rebuild is forced
            (without this, the next turn would just reload the stale prompt
            from SQLite — see ZET-1139).
          - ``GatewayRunner.invalidate_all_cached_agents()`` drops the
            in-process ``_cached_system_prompt`` on every cached AIAgent so
            the next turn rebuilds it fresh.

        The four steps together guarantee skill changes land on the very next
        turn for every session. The first two are skill-specific; the last
        two are the same hot-reload pair ``/v1/profile/reload`` uses for
        SOUL/IDENTITY/profile-text changes (skills also live in the system
        prompt, so the same invalidation applies).

        Auth: ZET_AGENT_KEY Bearer (same as chat).

        Response: ``{"cleared": true, "skills_total": <int>,
                     "invalidated_sessions": <int>, "db_rows_cleared": <int>}``.
        The first two fields are stable for older local-server callers; the
        last two are new in ZET-1139 and may be ignored by old callers.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache
            from agent.skill_commands import scan_skill_commands
        except Exception as exc:
            logger.exception("[zet_agent] skills-reload: import failed")
            return web.json_response(
                _openai_error(
                    f"skills reload modules unavailable: {exc}",
                    err_type="server_error",
                ),
                status=500,
            )

        try:
            clear_skills_system_prompt_cache(clear_snapshot=True)
            skill_commands = scan_skill_commands()
            skills_total = len(skill_commands)
        except Exception as exc:
            logger.exception("[zet_agent] skills-reload failed")
            return web.json_response(
                _openai_error(
                    f"skills reload failed: {exc}", err_type="server_error",
                ),
                status=500,
            )

        # ZET-1139 — also push the invalidation through to existing sessions.
        # Skills appear inside the system prompt, so the same SessionDB-clear
        # + cached-agent-invalidate pair used by /v1/profile/reload applies,
        # with the same asymmetric failure semantics:
        #
        #   - DB clear is CRITICAL: without it continuing sessions keep
        #     replaying the stored prompt (without the new skill) from
        #     SQLite. Returning 200 here while DB clear silently failed
        #     would re-introduce the ZET-1139 regression for skill changes.
        #     So a DB clear failure (or a missing SessionDB / gateway_runner)
        #     returns 500.
        #   - in-process invalidate is fail-soft: with DB already cleared,
        #     the next turn rebuilds from the cleared DB regardless of
        #     whether we managed to drop the in-process cache too.
        gw = getattr(self, "gateway_runner", None)
        if gw is None:
            logger.error("[zet_agent] skills-reload: no gateway_runner")
            return web.json_response(
                _openai_error(
                    "skills reload incomplete: no gateway runner",
                    err_type="server_error",
                ),
                status=500,
            )
        if _request_value(request, "hermes_profile_home"):
            session_db = self._ensure_session_db()
        else:
            session_db = getattr(gw, "_session_db", None)
        if session_db is None:
            logger.error("[zet_agent] skills-reload: no SessionDB on runner")
            return web.json_response(
                _openai_error(
                    "skills reload incomplete: no session db on runner",
                    err_type="server_error",
                ),
                status=500,
            )
        try:
            db_rows_cleared = session_db.clear_all_system_prompts()
        except Exception as exc:
            logger.exception(
                "[zet_agent] skills-reload: DB clear failed; "
                "returning 500 so caller can fall back",
            )
            return web.json_response(
                _openai_error(
                    f"skills reload db clear failed: {exc}",
                    err_type="server_error",
                ),
                status=500,
            )

        invalidated = 0
        try:
            invalidated = gw.invalidate_all_cached_agents()
        except Exception:
            logger.warning(
                "[zet_agent] skills-reload: invalidate-all failed "
                "(DB cleared, sessions still rebuild next turn)",
                exc_info=True,
            )

        logger.info(
            "[zet_agent] skills-reload: cleared prompt cache + rescanned "
            "(%d skill(s)); %d session(s) invalidated, %d DB row(s) cleared",
            skills_total, invalidated, db_rows_cleared,
        )
        return web.json_response({
            "cleared": True,
            "skills_total": skills_total,
            "invalidated_sessions": invalidated,
            "db_rows_cleared": db_rows_cleared,
        })

    async def _handle_connectors_reload(self, request: "web.Request") -> "web.Response":
        """POST /v1/connectors/reload — reconnect ONLY the zettlab_connectors
        MCP server and re-pull its tool table.

        Called by zettlab-local-server after installing/uninstalling a
        connector-skill (ZET-900 plan §4.2). The connector MCP server's
        ``tools/list`` is policy-gated and dynamic: a tool only appears if the
        agent's connector policy allows it. The tool table is frozen when the
        gateway first connects, so a policy change made after start-up is
        invisible until we explicitly reconnect this server.

        We reconnect ONLY ``zettlab_connectors`` — never a full
        ``shutdown_mcp_servers()`` — to avoid disturbing any other MCP server
        the agent has connected (a full shutdown also stops the whole MCP
        background loop). The single-server reconnect lives in
        ``tools/mcp_tool.reload_single_mcp_server`` (tear down just that server
        via its own ``shutdown()`` which self-deregisters its tools, then
        ``discover_mcp_tools()`` to reconnect the now-missing one).

        Runs the (blocking) reconnect in an executor so the aiohttp event loop
        is not stalled while the MCP handshake happens on its background loop.

        This only serves *new* sessions — it does not invalidate the current
        session's cached agent. That matches the skills-reload boundary.

        Auth: ZET_AGENT_KEY Bearer (same as chat).

        Response: ``{"reloaded": true, "tools_total": <int>}`` where
        ``tools_total`` is the count of ALL registered MCP tools (across every
        connected server) after the reconnect.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from tools.mcp_tool import reload_single_mcp_server
        except Exception as exc:
            logger.exception("[zet_agent] connectors-reload: import failed")
            return web.json_response(
                _openai_error(
                    f"mcp reload module unavailable: {exc}",
                    err_type="server_error",
                ),
                status=500,
            )

        import asyncio
        loop = asyncio.get_running_loop()
        try:
            tools = await loop.run_in_executor(
                None,
                reload_single_mcp_server,
                ZETTLAB_CONNECTORS_SERVER_NAME,
            )
        except Exception as exc:
            logger.warning(
                "[zet_agent] connectors-reload failed for server '%s': %s",
                ZETTLAB_CONNECTORS_SERVER_NAME, exc,
            )
            return web.json_response(
                _openai_error(
                    f"connector reload failed: {exc}", err_type="server_error",
                ),
                status=500,
            )

        tools_total = len(tools or [])
        logger.info(
            "[zet_agent] connectors-reload: reconnected '%s'; %d MCP tool(s) total",
            ZETTLAB_CONNECTORS_SERVER_NAME, tools_total,
        )
        return web.json_response({
            "reloaded": True,
            "tools_total": tools_total,
        })

    async def _handle_profile_reload(self, request: "web.Request") -> "web.Response":
        """POST /v1/profile/reload — make a profile-text change (SOUL.md,
        IDENTITY.md, agent name/description, in-prompt context files) take
        effect on the next turn of every session, including ones currently
        chatting.

        Called by zettlab-local-server after a config write that mutates a
        prompt-class file on disk. Without this endpoint, the change would
        be invisible to existing sessions until the user manually starts a
        new conversation (the continuing-session rebuild path re-uses the
        stored system_prompt to preserve the Anthropic prefix-cache prefix
        across turns).

        Two-step hot reload (semantics deliberately asymmetric):
          1. ``SessionDB.clear_all_system_prompts()`` nulls every session's
             stored prompt so the continuing-session rebuild is forced.
             **Critical** — without this the change is invisible to old
             sessions; failure here returns 500 so the local-server caller
             falls through to ``registry.Stop`` (lazy-respawn picks up the
             new file). The whole ZET-1139 hot-reload value depends on this
             succeeding, so swallowing the error and returning 200 would
             silently regress the very bug we're fixing.
          2. ``GatewayRunner.invalidate_all_cached_agents()`` clears the
             in-process ``_cached_system_prompt`` on every cached AIAgent
             so the next turn rebuilds from disk (re-runs SOUL.md /
             IDENTITY.md / context-files / memory loaders). **fail-soft**
             — step 1 already covers correctness (in-process cache is
             rebuilt from the now-cleared DB), so a failure here is logged
             and reported in the response count without poisoning the
             status code.

        Distinct from ``/v1/skills/reload`` (also extends to existing
        sessions now, but additionally clears the skills LRU + on-disk
        snapshot) and ``/v1/connectors/reload`` (MCP server reconnect, not
        a prompt change). Use this endpoint when only the prompt text on
        disk changed.

        Auth: ZET_AGENT_KEY Bearer (same as chat).

        Response (200): ``{"reloaded": true, "invalidated_sessions": <int>,
                           "db_rows_cleared": <int>}``.
        Response (500): when the critical DB-clear step fails (no
        SessionDB, write exception). Body is the standard OpenAI error
        envelope so callers can surface the reason. local-server then
        falls back to ``registry.Stop`` (see
        ``internal/agent/lifecycle/lifecycle.go``).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        gw = getattr(self, "gateway_runner", None)
        if gw is None:
            logger.error("[zet_agent] profile-reload: no gateway_runner")
            return web.json_response(
                _openai_error(
                    "profile reload unavailable: no gateway runner",
                    err_type="server_error",
                ),
                status=500,
            )

        # Step 1 (CRITICAL): clear DB-stored prompts. Without this the
        # continuing-session rebuild path keeps replaying the old prompt
        # from SQLite — exactly the ZET-1139 regression we're fixing.
        # Any failure here means we cannot keep our hot-reload contract,
        # so return 500 to let the local-server caller fall back to
        # ``registry.Stop`` (lazy respawn reads the new file fresh).
        if _request_value(request, "hermes_profile_home"):
            session_db = self._ensure_session_db()
        else:
            session_db = getattr(gw, "_session_db", None)
        if session_db is None:
            logger.error("[zet_agent] profile-reload: no SessionDB on runner")
            return web.json_response(
                _openai_error(
                    "profile reload unavailable: no session db on runner",
                    err_type="server_error",
                ),
                status=500,
            )
        try:
            db_rows_cleared = session_db.clear_all_system_prompts()
        except Exception as exc:
            logger.exception(
                "[zet_agent] profile-reload: DB clear failed; "
                "returning 500 so local-server falls back to Stop",
            )
            return web.json_response(
                _openai_error(
                    f"profile reload db clear failed: {exc}",
                    err_type="server_error",
                ),
                status=500,
            )

        # Step 2 (FAIL-SOFT): drop in-process cached prompts so existing
        # sessions rebuild on the next turn. Without this the next turn
        # would still rebuild from the cleared DB anyway — just one turn
        # later than ideal — so a failure here is not worth tearing down
        # the gateway over.
        profile = _request_value(request, "hermes_profile")
        invalidated = 0
        try:
            if profile and hasattr(gw, "invalidate_cached_agents_for_profile"):
                invalidated = gw.invalidate_cached_agents_for_profile(profile)
            else:
                invalidated = gw.invalidate_all_cached_agents()
        except Exception:
            logger.warning(
                "[zet_agent] profile-reload: invalidate failed "
                "(DB cleared, sessions still rebuild next turn)",
                exc_info=True,
            )

        logger.info(
            "[zet_agent] profile-reload: %d session(s) invalidated, "
            "%d DB row(s) cleared",
            invalidated, db_rows_cleared,
        )
        return web.json_response({
            "reloaded": True,
            "invalidated_sessions": invalidated,
            "db_rows_cleared": db_rows_cleared,
        })

    async def _handle_runtime_reset(self, request: "web.Request") -> "web.Response":
        """POST /v1/runtime/reset — reset one profile's in-memory runtime view.

        In multiplex mode local-server uses this for the user-facing
        "restart agent" action. Reuse the profile-reload invalidation path:
        continuing sessions drop stored prompts and cached AIAgent instances
        are evicted before the next turn. The shared gateway process stays up.
        """
        return await self._handle_profile_reload(request)

    async def _handle_profile_unload(self, request: "web.Request") -> "web.Response":
        """POST /v1/profile/unload — release cached state for one profile.

        local-server calls this before deleting the profile directory. The
        endpoint is deliberately best-effort: it never deletes files and only
        releases in-process caches owned by this adapter / runner.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        profile_home = _request_value(request, "hermes_profile_home")
        active_api_runs = self._active_profile_chat_runs(profile_home)
        if active_api_runs:
            return web.json_response(
                {
                    "unloaded": False,
                    "error": "profile has active sessions",
                    "active_sessions": active_api_runs,
                    "active_api_runs": active_api_runs,
                },
                status=409,
            )

        runtime_unload = {}
        gw = getattr(self, "gateway_runner", None)
        if gw is not None:
            try:
                profile = _request_value(request, "hermes_profile")
                unload = getattr(gw, "unload_profile_runtime", None)
                if callable(unload):
                    runtime_unload = await unload(profile)
                else:
                    runtime_unload = {
                        "evicted_sessions": gw.invalidate_all_cached_agents(),
                        "disconnected_adapters": 0,
                    }
            except Exception:
                logger.warning(
                    "[zet_agent] profile-unload: runtime unload failed",
                    exc_info=True,
                )
                return web.json_response(
                    _openai_error(
                        "profile unload failed: runtime state could not be released",
                        err_type="server_error",
                    ),
                    status=500,
                )

        if runtime_unload.get("blocked"):
            return web.json_response(
                {
                    "unloaded": False,
                    "error": "profile has active sessions",
                    "active_sessions": int(runtime_unload.get("active_sessions", 0) or 0),
                },
                status=409,
            )

        closed_session_db = False
        if profile_home:
            db = self._session_dbs.pop(self._profile_home_key(profile_home), None)
            if db is not None:
                close = getattr(db, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.warning(
                            "[zet_agent] profile-unload: SessionDB close failed",
                            exc_info=True,
                        )
                closed_session_db = True
            # 该 profile 的 goal barrier timers 一并取消（codex P1）：daemon
            # Timer 携带旧 profile 的 runtime scope，卸载后触发会用内存旧
            # scope 读 goal 并重新自驱一个用户刚删掉的 agent。getattr：
            # teardown 期间绝不懒创建 driver。
            drv = getattr(self, "_zet_goal_driver", None)
            if drv is not None:
                try:
                    drv.cancel_barrier_timers_for_home(profile_home)
                except Exception:
                    logger.warning(
                        "[zet_agent] profile-unload: goal timer cleanup failed",
                        exc_info=True,
                    )

        return web.json_response({
            "unloaded": True,
            "closed_session_db": closed_session_db,
            "evicted_sessions": int(runtime_unload.get("evicted_sessions", 0) or 0),
            "disconnected_adapters": int(runtime_unload.get("disconnected_adapters", 0) or 0),
        })

    # ------------------------------------------------------------------
    # connect — extend base routes with our respond endpoints
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the aiohttp server, registering our extra routes
        before the base class sets up the runner (which freezes the
        router).

        ``is_reconnect`` is accepted to preserve the BasePlatformAdapter
        connect contract; zet_agent does not need different cold-start versus
        reconnect behavior.

        We monkey-patch ``self._app.router`` immediately after the
        base method has built the application but before the freeze
        — by overriding ``_register_routes`` if it existed; since the
        base class inlines route registration in ``connect``, we use
        an ``on_app_built`` strategy: copy the base body verbatim is
        too much, so we wrap by *appending* routes through a startup
        signal that fires before freeze.

        aiohttp finalises routes during ``Application.startup`` (which
        is invoked by ``runner.setup()``). Adding routes after that
        raises ``RuntimeError: Cannot register a route once the
        application has started``. So we hook the ``on_startup`` list
        with the highest-priority callback that adds our routes
        *before* startup completes — but on_startup runs *during*
        startup, after freezing. The robust path is: install our
        routes pre-startup by re-defining connect to splice them in.

        Concretely, we run super().connect() inside an asyncio task
        wrapper that, before yielding to the base method, calls
        ``self._app = web.Application(...)`` ourselves, registers
        OUR routes, and then lets base's connect rebuild on top.
        That's brittle. The clean path is to override connect()
        and copy the base body — which we do here, with the four
        extra ``add_post`` calls inlined.
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        # Pull the helpers we need from the base module. They're
        # private to the module but importable since this is the
        # same package layer.
        from gateway.platforms.api_server import (
            cors_middleware,
            body_limit_middleware,
            security_headers_middleware,
            is_network_accessible,
        )
        import asyncio
        import socket as _socket

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
            # client_max_size=MAX_REQUEST_BYTES mirrors APIServerAdapter.connect
            # in api_server.py — without it aiohttp falls back to its 1 MiB
            # default and rejects multimodal payloads (image_url with inlined
            # base64) before they reach _handle_chat_completions, surfacing
            # as a misleading 400 "Invalid JSON in request body".
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            self._app["api_server_adapter"] = self
            # Base routes — kept identical to APIServerAdapter.connect
            # so health/models/responses/runs/jobs all work under the
            # zet_agent platform too.
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            if hasattr(self, "_handle_capabilities"):
                self._app.router.add_get("/v1/capabilities", self._handle_capabilities)
            self._app.router.add_post("/v1/chat/completions", self._diagnostic_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            if hasattr(self, "_handle_get_run"):
                self._app.router.add_get("/v1/runs/{run_id}", self._handle_get_run)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)

            # ZET-372 — interaction respond endpoints.
            self._app.router.add_post(
                "/v1/sessions/{session_id}/approval/respond",
                self._handle_approval_respond,
            )
            self._app.router.add_post(
                "/v1/sessions/{session_id}/clarify/respond",
                self._handle_clarify_respond,
            )
            self._app.router.add_get(
                "/v1/sessions/{session_id}/pending",
                self._handle_pending,
            )
            self._app.router.add_post(
                "/v1/sessions/{session_id}/interrupt",
                self._handle_session_interrupt,
            )
            # Persistent goal loop control surface (create/pause/resume/clear
            # /status) — consumed by zettlab-local-server only.
            self._app.router.add_post(
                "/v1/sessions/{session_id}/goal",
                self._handle_session_goal,
            )
            self._app.router.add_get(
                "/v1/sessions/{session_id}/goal",
                self._handle_session_goal,
            )
            self._app.router.add_post(
                "/v1/model/switch",
                self._handle_model_switch,
            )
            self._app.router.add_post(
                "/v1/sessions/{session_id}/model/switch",
                self._handle_session_model_switch,
            )
            self._app.router.add_delete(
                "/v1/sessions/{session_id}/model",
                self._handle_session_model_clear,
            )

            # ZET-900 — skill / connector reload control endpoints.
            # Triggered by zettlab-local-server after a skillhub install /
            # uninstall lands so the running gateway picks up new skills
            # (prompt cache) and connector tools (zettlab_connectors MCP)
            # without a cold restart. Auth: ZET_AGENT_KEY Bearer.
            self._app.router.add_post(
                "/v1/skills/reload",
                self._handle_skills_reload,
            )
            self._app.router.add_post(
                "/v1/connectors/reload",
                self._handle_connectors_reload,
            )
            # ZET-1139 — hot reload for profile-text changes (SOUL.md,
            # IDENTITY.md, agent name/description, in-prompt context
            # files). Triggered by zettlab-local-server after a config
            # write that mutates a prompt-class file on disk.
            self._app.router.add_post(
                "/v1/profile/reload",
                self._handle_profile_reload,
            )
            self._app.router.add_post(
                "/v1/runtime/reset",
                self._handle_runtime_reset,
            )
            self._app.router.add_post(
                "/v1/profile/unload",
                self._handle_profile_unload,
            )
            self._register_profile_api_routes(
                self._app.router,
                chat_handler=self._diagnostic_chat_completions,
            )
            self._app.router.add_post(
                "/p/{profile}/v1/skills/reload",
                self._profile_handler(self._handle_skills_reload),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/connectors/reload",
                self._profile_handler(self._handle_connectors_reload),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/profile/reload",
                self._profile_handler(self._handle_profile_reload),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/runtime/reset",
                self._profile_handler(self._handle_runtime_reset),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/profile/unload",
                self._profile_handler(self._handle_profile_unload),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/model/switch",
                self._profile_handler(self._handle_model_switch),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/model/switch",
                self._profile_handler(self._handle_session_model_switch),
            )
            self._app.router.add_delete(
                "/p/{profile}/v1/sessions/{session_id}/model",
                self._profile_handler(self._handle_session_model_clear),
            )
            self._app.router.add_get(
                "/p/{profile}/v1/sessions/{session_id}/pending",
                self._profile_handler(self._handle_pending),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/approval/respond",
                self._profile_handler(self._handle_approval_respond),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/clarify/respond",
                self._profile_handler(self._handle_clarify_respond),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/interrupt",
                self._profile_handler(self._handle_session_interrupt),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/goal",
                self._profile_handler(self._handle_session_goal),
            )
            self._app.router.add_get(
                "/p/{profile}/v1/sessions/{session_id}/goal",
                self._profile_handler(self._handle_session_goal),
            )

            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires ZET_AGENT_KEY/API_SERVER_KEY.",
                    self.name, self._host,
                )
                return False

            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error(
                            "[%s] Refusing to start: API key looks like a placeholder. "
                            "Set ZET_AGENT_KEY to a real secret.",
                            self.name,
                        )
                        return False
                except ImportError:
                    pass

            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(("127.0.0.1", self._port))
                logger.error(
                    "[%s] Port %d already in use. Set ZET_AGENT_PORT to a different port.",
                    self.name, self._port,
                )
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            logger.info(
                "[%s] listening on http://%s:%d (interaction endpoints enabled)",
                self.name, self._host, self._port,
            )

            # Goal reconcile-on-start: after a crash/OOM respawn (local-server
            # goal keepalive re-spawns us), re-report every indexed goal and
            # re-kick loops that were cut mid-flight (HR#2 self-heal pair).
            # 必须在 API key / 端口检查和 site.start() 全部成功之后才创建
            # （codex P1）：启动失败的副本（端口被占等）若也自驱 goal，会和
            # 真正监听的进程并发重踢同一循环。
            goal_task = asyncio.create_task(self._goals().reconcile_on_start())
            try:
                self._background_tasks.add(goal_task)
            except TypeError:
                pass
            if hasattr(goal_task, "add_done_callback"):
                goal_task.add_done_callback(self._background_tasks.discard)

            return True
        except Exception:
            logger.exception("[%s] failed to start", self.name)
            return False

    async def disconnect(self) -> None:
        """Tear down the aiohttp server and unregister approval callbacks
        for any sessions we registered."""
        # Goal barrier timers are daemon threading.Timers OUTSIDE
        # _background_tasks — cancel them here or a reloaded/replaced
        # adapter's stale timers keep firing wakeups and double-drive the
        # goal alongside the new adapter's reconcile (codex P1). getattr:
        # never lazily CREATE the driver during teardown.
        drv = getattr(self, "_zet_goal_driver", None)
        if drv is not None:
            try:
                drv.cancel_all_barrier_timers()
            except Exception:
                pass

        # Drop approval notify callbacks so blocked agent threads (if
        # any leak past process shutdown) don't fire into a dead loop.
        try:
            from tools.approval import unregister_gateway_notify
            with self._session_lock:
                sids = list(self._session_titles.keys())
            for sid in sids:
                try:
                    unregister_gateway_notify(sid)
                except Exception:
                    pass
        except Exception:
            pass

        # Wake any clarify waiters with empty responses so the agent
        # threads don't sit on threading.Event forever.
        with self._clarify_state_lock:
            entries = []
            for queue in self._clarify_queues.values():
                entries.extend(queue)
            self._clarify_queues.clear()
        for entry in entries:
            entry.event.set()

        await super().disconnect()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if metadata and metadata.get("zet_agent_cron_delivery"):
            # cron 投递实际由 zet_agent_cron monkey-patch 完成（写 SessionDB +
            # POST ZET_CHAT_APPEND_URL）；这里只给 scheduler 的显式内部调用
            # 返回 success，避免普通 send_message 被静默吞掉。
            return SendResult(success=True, message_id=f"zet_agent:{chat_id}")
        return SendResult(success=False, error="zet_agent has no proactive send path")
