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
After the first successful exchange, reuse Hermes' native
``agent.title_generator`` LLM worker and emit its result as a
``conversation.title`` event before the request SSE closes.
"""

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
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


async def _to_thread_with_completion_barrier(func, /, *args, **kwargs):
    """Keep a cancelled request alive until its non-cancellable worker exits.

    ``asyncio.to_thread`` cancellation only cancels the asyncio wrapper. The
    underlying thread keeps mutating profile state, so callers must not release
    an unload barrier until that worker has actually finished. Cancellation is
    still re-raised after the worker result/exception has been observed.
    """
    worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # A second cancellation request must not reopen the same race.
                continue
            except BaseException:
                # The cancelled request must keep cancellation as its public
                # outcome even when the worker finishes with an exception.
                break
        try:
            worker.result()
        except BaseException:
            # Cancellation remains the externally visible outcome, but consume
            # the worker exception so asyncio does not report it as unhandled.
            pass
        raise cancelled


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

# Expired runtime-import staging may contain complete source transcripts.
# Sweep periodically even when no later import request arrives; each database
# call deletes one bounded batch to avoid long write-lock holds on device.
RUNTIME_IMPORT_CLEANUP_INTERVAL_SECONDS = 15 * 60

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
_ZET_ADDENDUM_HEAD = """\
## 工作风格

执行以下结构化变更前，先用 clarify 工具向用户确认意图（把关键参数列成 2-4 个选项让用户选）：
- 创建 / 修改 / 删除定时任务
- 删除数据、清空记录、批量操作
- 发送外部消息（邮件、IM 推送）

用户已经明确指定全部关键参数（频率、时间、目标、内容）时直接执行，无需再 clarify。
信息查询、闲聊、回答问题不要 clarify。"""

# Plan-First section, auto-execute variant: App opted in (capability negotiation)
# to render the plan card as a read-only preview and let the model carry the plan
# out in the same turn.
_ZET_PLAN_FIRST_AUTO = """\
## 计划先行（Plan-First）

面对复杂多步任务（涉及 3 个以上阶段、不可逆操作或大量数据变更）时：
1. 先调用 `present_plan` 工具，把执行计划结构化呈现给用户（分组列出每步要做什么）。
2. 计划卡片只是给用户看的只读预览，展示后**不要停下、不要等用户确认、不要问用户是否执行**，直接在同一轮继续把计划执行下去。
3. 执行阶段用 `todo` 工具逐步记录和更新进度，每完成一步立即把对应 todo 标记为 completed。

用户说"plan 模式"、"计划模式"、"先给计划"时，也按上述 App 计划卡片流程处理（展示计划后直接执行，不等确认）。
不要加载名为 `plan` 的 markdown skill，也不要写 `.hermes/plans`；那是 CLI/文档计划模式，不是 Zettlab App 的确认卡片。

简单的单步请求、查询、闲聊不需要 present_plan，直接执行即可。"""

# Plan-First section, manual variant: no auto-execute opt-in (legacy confirm card,
# and the safe default for any client that did not opt in). Mirrors the global
# PLAN_SCHEMA stop-and-wait contract so a self-initiated present_plan (outside App
# plan mode, where _should_end_after_present_plan does NOT halt the turn) cannot
# run write/terminal/message side effects before the user confirms.
_ZET_PLAN_FIRST_MANUAL = """\
## 计划先行（Plan-First）

面对复杂多步任务（涉及 3 个以上阶段、不可逆操作或大量数据变更）时：
1. 先调用 `present_plan` 工具，把执行计划结构化呈现给用户（分组列出每步要做什么）。
2. 计划卡片是给用户确认的预览，展示后**停下、等用户在确认卡上确认后再执行**；在收到用户确认前，不要执行计划里的任何实际操作（写文件、terminal、发送外部消息等有副作用的动作）。
3. 收到用户确认后再逐步执行，用 `todo` 工具记录和更新进度，每完成一步立即把对应 todo 标记为 completed。

用户说"plan 模式"、"计划模式"、"先给计划"时，也按上述 App 计划卡片流程处理（展示计划后停下，等用户确认再执行）。
不要加载名为 `plan` 的 markdown skill，也不要写 `.hermes/plans`；那是 CLI/文档计划模式，不是 Zettlab App 的确认卡片。

简单的单步请求、查询、闲聊不需要 present_plan，直接执行即可。"""

_ZET_ADDENDUM_TAIL = """\
## 用户画像语言

写入长期用户画像（memory 工具 target="user"，即 USER.md）时，必须使用简体中文。
姓名、产品名、命令、代码标识符可以保留原文，但描述用户特征、偏好、沟通风格的正文必须写成中文。"""


def _zettlab_workflow_addendum(auto_execute: bool) -> str:
    """Assemble the zet_agent workflow addendum with a capability-aware Plan-First
    section.

    ``auto_execute`` mirrors the resolved per-turn ``_zet_agent_plan_auto_execute``
    flag (App capability opt-in > env kill-switch > default False). When True the
    Plan-First section tells the model to carry the plan out in the same turn after
    ``present_plan``; when False it tells the model to stop and wait for the user's
    confirmation, matching the legacy confirm card and the global stop-and-wait
    PLAN_SCHEMA so no side effect runs before the user confirms.
    """
    plan_first = _ZET_PLAN_FIRST_AUTO if auto_execute else _ZET_PLAN_FIRST_MANUAL
    return "\n\n".join((_ZET_ADDENDUM_HEAD, plan_first, _ZET_ADDENDUM_TAIL)) + "\n"


_DELEGATION_ADVANCE_ENV = "ZET_DELEGATION_ADVANCE_URL"


def _delegation_advance_url() -> str:
    """Resolve local-server's loopback delegation-advance endpoint.

    Mirrors ZET_GOAL_ADVANCE_URL resolution: profile ``.env`` first so
    multiplex profiles stay authoritative, then process env. The URL is
    device-global (one local-server per device), so the os.environ fallback
    cannot cross profiles the way a per-profile secret could.
    """
    try:
        url = _zet_agent_cron._scoped_env(_DELEGATION_ADVANCE_ENV, "").strip()
    except Exception:
        url = ""
    if url:
        return url
    return os.environ.get(_DELEGATION_ADVANCE_ENV, "").strip()


def check_zet_agent_requirements() -> bool:
    """Return True iff this platform can be started in the current process."""
    return AIOHTTP_AVAILABLE


class _ClarifyEntry:
    """One pending clarify request inside a session FIFO queue.

    The agent thread enters ``wait()`` on ``event``; the HTTP respond
    handler resolves the matching entry, stores the response, and calls
    ``event.set()`` to unblock the agent. ``clarify_id`` is generated at
    creation and is the stable identity sent both over the live stream and
    through the reconnect ``/pending`` projection. Older callers may omit it
    when responding, in which case the historical FIFO behavior is retained.
    """

    __slots__ = ("clarify_id", "event", "payload", "response")

    def __init__(self, clarify_id: str, payload: Dict[str, Any]) -> None:
        self.clarify_id = clarify_id
        self.event = threading.Event()
        self.payload = payload
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

        # Pending clarify prompts: {profile-home}|{session_id} ->
        # list[_ClarifyEntry] (FIFO). A bare session id is not a gateway
        # identity in multiplex mode: /p/main and /p/coder may legitimately
        # run one same-named session at the same time.
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
        # payload per scoped session. Used by GET /v1/sessions/{sid}/pending
        # so a reconnecting client (chat.resume path) can re-render the modal
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

        # Sessions with registered approval callbacks. Kept separately from
        # title generation so titles remain fully owned by Hermes SessionDB.
        self._session_lock = threading.Lock()
        self._approval_session_ids: set[str] = set()

        # Portable imports mutate profile-owned state outside the chat-run
        # path. Track them explicitly so a successful profile unload cannot
        # race a worker that later recreates the deleted profile directory.
        self._runtime_import_operation_lock = threading.Lock()
        self._runtime_import_operations: Dict[str, int] = {}
        self._runtime_import_unload_barriers: Dict[
            str, Dict[int, tuple[Optional[tuple[int, int]], bool]]
        ] = {}
        self._runtime_import_barrier_generation = 0

    @staticmethod
    def _profile_directory_identity(key: str) -> Optional[tuple[int, int]]:
        try:
            stat = Path(key).stat()
            return int(stat.st_dev), int(stat.st_ino)
        except OSError:
            return None

    def _begin_runtime_import_operation(self, profile_home: Optional[Any]) -> Optional[str]:
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            if self._runtime_import_barriers_locked(key):
                return None
            self._runtime_import_operations[key] = (
                self._runtime_import_operations.get(key, 0) + 1
            )
        return key

    def _end_runtime_import_operation(self, key: Optional[str]) -> None:
        if not key:
            return
        with self._runtime_import_operation_lock:
            remaining = self._runtime_import_operations.get(key, 0) - 1
            if remaining > 0:
                self._runtime_import_operations[key] = remaining
            else:
                self._runtime_import_operations.pop(key, None)

    def _runtime_import_barriers_locked(
        self, key: str
    ) -> Dict[int, tuple[Optional[tuple[int, int]], bool]]:
        barriers = self._runtime_import_unload_barriers.get(key)
        if not barriers:
            return {}
        current = self._profile_directory_identity(key)
        # An absent directory remains blocked: otherwise an import request can
        # recreate the profile it is meant to protect. A different live inode
        # is a new profile generation, so only owners acquired for that exact
        # generation continue to apply.
        if current is None:
            return barriers
        retained = {
            owner: state
            for owner, state in barriers.items()
            if state[0] == current
        }
        if retained:
            self._runtime_import_unload_barriers[key] = retained
        else:
            self._runtime_import_unload_barriers.pop(key, None)
        return retained

    def _block_runtime_import_profile(
        self, profile_home: Optional[Any]
    ) -> tuple[int, Optional[int]]:
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            active = int(self._runtime_import_operations.get(key, 0) or 0)
            if active:
                return active, None
            self._runtime_import_barriers_locked(key)
            self._runtime_import_barrier_generation += 1
            owner = self._runtime_import_barrier_generation
            self._runtime_import_unload_barriers.setdefault(key, {})[owner] = (
                self._profile_directory_identity(key), False
            )
            return 0, owner

    def _complete_runtime_import_profile_unload(
        self, profile_home: Optional[Any], owner: Optional[int]
    ) -> None:
        if owner is None:
            return
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            barriers = self._runtime_import_unload_barriers.get(key)
            if barriers is None or owner not in barriers:
                return
            identity, _pending = barriers[owner]
            # Completed owners collapse to the newest generation. Pending
            # owners remain individually reference-counted so one failed
            # request can release only itself without growing the durable
            # successful barrier set on repeated unload calls.
            for previous_owner, (_previous_identity, completed) in tuple(
                barriers.items()
            ):
                if completed:
                    barriers.pop(previous_owner, None)
            barriers[owner] = (identity, True)

    def _unblock_runtime_import_profile(
        self, profile_home: Optional[Any], owner: Optional[int]
    ) -> None:
        if owner is None:
            return
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            barriers = self._runtime_import_unload_barriers.get(key)
            if barriers is None:
                return
            barriers.pop(owner, None)
            if not barriers:
                self._runtime_import_unload_barriers.pop(key, None)

    def _snapshot_runtime_import_reload_barriers(
        self, profile_home: Optional[Any]
    ) -> tuple[str, frozenset[int]]:
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            barriers = self._runtime_import_barriers_locked(key)
            return key, frozenset(
                owner for owner, (_identity, completed) in barriers.items()
                if completed
            )

    def _release_runtime_import_reload_barriers(
        self, key: str, owners: frozenset[int]
    ) -> None:
        if not owners:
            return
        with self._runtime_import_operation_lock:
            barriers = self._runtime_import_unload_barriers.get(key)
            if barriers is None:
                return
            for owner in owners:
                barriers.pop(owner, None)
            if not barriers:
                self._runtime_import_unload_barriers.pop(key, None)

    def _runtime_import_profile_is_blocked(self, profile_home: Optional[Any]) -> bool:
        key = self._profile_home_key(profile_home)
        with self._runtime_import_operation_lock:
            return bool(self._runtime_import_barriers_locked(key))

    # ------------------------------------------------------------------
    # Async delegation delivery (delegate_task background=true)
    # ------------------------------------------------------------------

    @property
    def supports_async_delivery(self) -> bool:  # type: ignore[override]
        """Background delegation is available only when local-server has
        published its delegation-advance endpoint (ZET_DELEGATION_ADVANCE_URL
        in the profile ``.env`` / process env). Without it delegate_task keeps
        the upstream synchronous fallback, so rollout order is safe: new
        hermes + old local-server behaves exactly like today.
        """
        return bool(_delegation_advance_url())

    def _bind_turn_session_context(self, session_id: str) -> None:
        """Rebind session contextvars for this turn's agent build.

        ZettClaw — 让 cronjob tool 自动设 origin: 把当前 chat session_id 注入
        contextvars，cronjob_tools._origin_from_env 会读到 platform/chat_id
        自动填到 cron job.origin。否则 cron 触发时 OriginStrategy 找不到 chat
        → 走 NewSession 兜底创 phantom session, APP 看不到推送。

        tokens 不显式 reset — contextvars 是 task-local，task 结束自动清；
        同 task 内多次 _create_agent 后 set 会覆盖前值，符合预期。

        async_delivery 必须显式传：这次 set 覆盖了上游
        _bind_api_server_session 刚写下的 False，而参数默认值是 True——不
        显式绑定 supports_async_delivery（env 门控）的话，
        ZET_DELEGATION_ADVANCE_URL 未配置的部署里 delegate_task 会承诺
        background 却无处投递完成事件（#10760 型 silent no-op，durable 行
        永远 claimable）；配置了则如实开闸。
        """
        if not session_id:
            return
        try:
            from gateway.session_context import set_session_vars

            set_session_vars(
                platform="zet_agent",
                chat_id=session_id,
                chat_name="",  # 暂留空，APP 这边的 chat title 不通过这条路径来
                thread_id="",
                user_id="",
                user_name="",
                session_key=session_id,
                async_delivery=self.supports_async_delivery,
            )
        except Exception as _e:
            logger.warning("[zet_agent] set_session_vars failed (cron origin won't auto-populate): %s", _e)

    def resolve_process_event_source(self, session_key: str):
        """Claim synthetic process events whose session_key is a zet_agent
        session id.

        zet_agent binds the local-server session id verbatim as the gateway
        session_key, so ``_build_process_event_source``'s generic
        ``platform:chat_type:chat_id`` parse never matches and async
        delegation completions would be dropped as unroutable. Ownership is
        verified against Hermes SessionDB (fail-closed): only sessions this
        gateway actually persisted are claimed, so foreign platforms' keys
        stay unresolvable.
        """
        key = (session_key or "").strip()
        if not key:
            return None
        try:
            db = self._ensure_session_db()
            if db is None or db.get_session(key) is None:
                return None
        except Exception as exc:
            # Probe ERROR ≠ "not ours". SQLite busy / briefly unreadable DB
            # must surface as transient so the delivery loop requeues the
            # event — folding it into None drops the completion from the
            # in-memory queue while its durable row is never rescanned here.
            logger.debug(
                "[zet_agent] session ownership probe failed for %s",
                key,
                exc_info=True,
            )
            from gateway.run import TransientRouteResolutionError

            raise TransientRouteResolutionError(
                f"session ownership probe failed for {key}"
            ) from exc
        from gateway.session import SessionSource

        return SessionSource(
            platform=Platform.ZET_AGENT,
            chat_id=key,
            chat_type="dm",
        )

    async def handle_message(self, event) -> None:
        """Divert internal async-delegation completions to local-server.

        Upstream's watcher forges a new internal turn via ``handle_message``
        and relies on the adapter's outbound send path for the reply — the
        api_server family has none, so that turn's output would be lost.
        Instead POST the structured completion to local-server's loopback
        delegation-advance endpoint; local-server starts a first-class turn
        on the originating session and the App receives a normally streamed
        reply. Raising is the retry signal: the watcher releases its durable
        claim and redelivers later.
        """
        process_event = None
        if getattr(event, "internal", False):
            meta = getattr(event, "metadata", None)
            if isinstance(meta, dict):
                pe = meta.get("process_event")
                if isinstance(pe, dict) and str(pe.get("type") or "") == "async_delegation":
                    process_event = pe
        if process_event is None:
            await super().handle_message(event)
            return
        await self._deliver_delegation_completion(
            process_event, str(getattr(event, "text", "") or "")
        )

    async def _deliver_delegation_completion(
        self, evt: Dict[str, Any], synth_text: str
    ) -> None:
        """POST one async-delegation completion to local-server.

        Contract with the watcher (``_deliver_completion_notification``):
        returning normally means "accepted" (the durable row is acked);
        raising means "retry later" (the claim is released). local-server's
        endpoint is loopback-only (mirrors /api/v1/internal/goal/advance),
        so no per-profile action token is attached.
        """
        url = _delegation_advance_url()
        if not url:
            # Unreachable in practice — supports_async_delivery gates dispatch
            # on the same env — but raise rather than ack a completion nobody
            # delivered; the durable row stays claimable for retry.
            raise RuntimeError(
                "ZET_DELEGATION_ADVANCE_URL unset; cannot deliver async "
                "delegation completion"
            )
        payload = {
            "schema": 1,
            "kind": "delegation",
            "session_key": str(evt.get("session_key") or ""),
            "session_id": str(evt.get("parent_session_id") or ""),
            "delegation_id": str(evt.get("delegation_id") or ""),
            "status": evt.get("status"),
            "goal": evt.get("goal"),
            "goals": evt.get("goals"),
            "is_batch": bool(evt.get("is_batch")),
            "results": evt.get("results"),
            "summary": evt.get("summary"),
            "error": evt.get("error"),
            "model": evt.get("model"),
            "role": evt.get("role"),
            "dispatched_at": evt.get("dispatched_at"),
            "completed_at": evt.get("completed_at"),
            "duration_seconds": evt.get("duration_seconds")
            or evt.get("total_duration_seconds"),
            "synth_text": synth_text,
        }

        def _post() -> int:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return int(getattr(resp, "status", 0) or 0)
            except urllib.error.HTTPError as http_exc:
                # Non-2xx is a RESPONSE, not a transport failure — surface the
                # status code so the caller can split permanent vs retryable.
                return int(http_exc.code)

        try:
            status = await asyncio.to_thread(_post)
        except Exception as exc:
            # Transport-level failure (connection refused / timeout): the
            # local-server may just be restarting — keep the durable row
            # claimable and let the watcher retry.
            raise RuntimeError(
                f"delegation-advance delivery failed for "
                f"{payload['delegation_id'] or '<no-id>'}: {exc}"
            ) from exc
        if 400 <= status < 500 and status not in (408, 429):
            # Permanent rejection (session deleted, payload judged invalid…):
            # retrying can never succeed — dead-letter by logging the full
            # identity and returning normally so the durable row is acked and
            # the 2s watcher loop stops re-posting it (also across restarts).
            logger.error(
                "[zet_agent] delegation-advance delivery permanently rejected "
                "with HTTP %d for %s (session_key=%s); dropping after "
                "dead-letter log",
                status,
                payload["delegation_id"] or "<no-id>",
                payload["session_key"] or "<none>",
            )
            return
        if not (200 <= status < 300):
            # 5xx / 408 / 429: server-side transient — retryable.
            raise RuntimeError(
                f"delegation-advance delivery rejected with HTTP {status} "
                f"for {payload['delegation_id'] or '<no-id>'}"
            )
        logger.info(
            "[zet_agent] delivered async delegation completion %s (session_key=%s)",
            payload["delegation_id"] or "<no-id>",
            payload["session_key"] or "<none>",
        )

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
    def _push_title(stream_q: Any, title: Optional[str]) -> None:
        if not title or stream_q is None:
            return
        try:
            # Hermes keeps the upstream payload minimal; local-server adds
            # the canonical session_id while translating the SSE event.
            stream_q.put((
                "__tool_progress__",
                {"type": "conversation.title", "title": title},
            ))
        except Exception:
            logger.debug("[zet_agent] title push failed", exc_info=True)

    @staticmethod
    def _title_user_message(user_message: str) -> str:
        """Remove local-server routing instructions from the title input."""
        text = str(user_message or "").lstrip()
        marker = "\n[User request]\n"
        if text.startswith("[Zettlab internal routing directive]") and marker in text:
            return text.rsplit(marker, 1)[1].strip()
        return text

    # Bounded concurrency for skill expansion: the worker threads come from
    # the SAME default executor _run_agent runs on, and expansion happens
    # BEFORE the request counts against _inflight_agent_runs — without its own
    # cap, a burst of skill-invocation requests could queue enough scan/load
    # jobs to starve real agent runs. Saturation fails open: the message
    # passes through unexpanded instead of queueing.
    _SKILL_INVOKE_MAX_CONCURRENCY = 4
    _SKILL_INVOKE_ACQUIRE_TIMEOUT = 2.0
    _skill_invoke_semaphore = None

    async def _expand_inbound_skill_invocation(
        self,
        user_message: Any,
        skill_slug: str,
        session_id: Optional[str] = None,
        on_settled: Optional[Any] = None,
    ) -> Any:
        """Async shell: fast-path pass-through, then expand off the event loop.

        The cheap shape checks run inline; anything that touches the skills
        layer (directory scan, SKILL.md load, template expansion) is blocking
        file I/O and must NOT run on the aiohttp event loop — a slow disk or a
        large external skills dir would stall every session's SSE / approval /
        reload traffic. ``asyncio.to_thread`` copies the current contextvars
        into the worker, so the platform binding inside the blocking body
        stays task-local.
        """
        def _settled() -> None:
            # 结算回调与信号量同生命周期:凡释放许可(或根本没占用)的点
            # 恰好一次地通知调用方「展开副作用已终止」。
            if on_settled is None:
                return
            try:
                on_settled()
            except Exception:
                logger.warning(
                    "[zet_agent] skill expansion on_settled callback failed",
                    exc_info=True,
                )

        if not skill_slug or not isinstance(user_message, str):
            _settled()
            return user_message
        import asyncio

        sema = self._skill_invoke_semaphore
        if sema is None:
            # Lazy init on the event loop; no await between check and set, so
            # concurrent first calls cannot race in a single-threaded loop.
            sema = asyncio.Semaphore(self._SKILL_INVOKE_MAX_CONCURRENCY)
            self._skill_invoke_semaphore = sema
        try:
            await asyncio.wait_for(
                sema.acquire(), timeout=self._SKILL_INVOKE_ACQUIRE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[zet_agent] skill expansion saturated; passing message through",
            )
            _settled()
            return user_message
        except asyncio.CancelledError:
            # 许可未取得、worker 不存在:副作用已终止,当场结算。
            _settled()
            raise
        # Permit accounting must survive BOTH cancellation shapes (a
        # try/finally or a done-callback on the asyncio wrapper handles
        # neither correctly):
        #   - worker RUNNING when the caller is cancelled: the thread keeps
        #     going, so the permit must stay held until the worker's own
        #     finally releases it (early release = connect-and-drop loop
        #     bypasses the cap and piles workers onto the shared executor);
        #   - worker still QUEUED when the caller is cancelled: the executor
        #     future is cancelled before the fn ever starts, its finally will
        #     never run, so the CALLER must refund the permit right here —
        #     otherwise 4 drops leak all permits until process restart.
        # A started/released flag pair under a lock makes the two paths
        # mutually exclusive and the release exactly-once; a worker that
        # loses the race (starts after a queued-cancel refund) exits without
        # touching the skills layer. run_in_executor + copied context
        # (to_thread equivalent) keeps the platform binding task-local.
        import contextvars

        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        state_lock = threading.Lock()
        state = {"started": False, "released": False}

        def _finish_on_loop():
            sema.release()
            _settled()

        def _release_from_worker():
            with state_lock:
                if state["released"]:
                    return
                state["released"] = True
            try:
                loop.call_soon_threadsafe(_finish_on_loop)
            except RuntimeError:
                # Loop already closed (shutdown) — the permit is moot.
                pass

        def _worker():
            with state_lock:
                if state["released"]:
                    # Queued-cancel already refunded the permit; stay out of
                    # the skills layer (the caller is gone anyway).
                    return user_message
                state["started"] = True
            try:
                return ctx.run(
                    self._expand_inbound_skill_invocation_blocking,
                    user_message, skill_slug, session_id,
                )
            finally:
                _release_from_worker()

        fut = loop.run_in_executor(None, _worker)
        try:
            return await fut
        except asyncio.CancelledError:
            with state_lock:
                refund = not state["started"] and not state["released"]
                if refund:
                    state["released"] = True
            if refund:
                # queued-cancel:fn 永不执行,当场退款并结算。
                sema.release()
                _settled()
            raise

    def _expand_inbound_skill_invocation_blocking(
        self, user_message: str, skill_slug: str, session_id: Optional[str] = None
    ) -> Any:
        """Expand an explicitly requested skill (metadata.skill_slug) into the
        full skill payload.

        The App's skill quick-pick inserts a visible ``/<slug>`` token into
        the input text AND sends ``metadata.skill_slug`` with the message; the
        client drops the field when the user edits the token away. Only that
        explicit field triggers expansion — the text is never sniffed for
        slash commands (in-band signaling is ambiguous: "/<skill> 是什么"
        would fire the skill). On the CLI the same expansion is done by the
        slash command layer; this hook gives the App path the same guarantee.

        Behavior contract (HR4 — pure addition, fail-open):
          - only fires when the slug resolves to an installed skill; an
            unknown/stale slug (App inventory drift) logs and passes the
            message through byte-identical.
          - the visible ``/<slug>`` token(s) the quick-pick inserted are
            stripped from the task text (they are display artifacts, not part
            of the user's instruction); everything else is preserved.
          - any internal failure logs and falls back to the original text —
            a broken skill must degrade to a plain message, never block it.

        Three invariants this hook must uphold:
          - the whole expansion runs with the platform contextvar bound to
            ``zet_agent`` (push/pop, token-restored): it executes in the HTTP
            handler BEFORE the session is bound, and without the binding
            ``skills.platform_disabled.zet_agent`` and frontmatter
            ``platforms:`` filters silently resolve against no platform —
            a skill disabled only for zet_agent would still expand.
          - the fork's skill-scope resolvers read the session ContextVar
            BEFORE the process ``HERMES_PLATFORM`` env (ZET fork semantic —
            see skill_commands._resolve_skill_commands_platform), so the
            binding above governs scan/build even when an external env value
            exists, without mutating process-global state that a co-hosted
            platform could observe. The platform-disabled gate is ALSO
            enforced with an explicit ``platform="zet_agent"`` argument —
            top precedence, cannot be shadowed by anything.
          - the payload is built by the canonical
            ``build_skill_invocation_message`` (same scaffolding as the CLI
            slash): MemoryManager._strip_skill_scaffolding keys off the
            canonical activation prefix to recover the user's instruction,
            so a bespoke note here would leak the full skill body into
            long-term memory / embeddings.
        """
        from gateway.session_context import (
            pop_session_platform,
            push_session_platform,
        )

        token = "/" + skill_slug
        platform_token = push_session_platform("zet_agent")
        try:
            return self._expand_under_platform_binding(
                user_message, skill_slug, token, session_id
            )
        finally:
            pop_session_platform(platform_token)

    def _expand_under_platform_binding(
        self,
        user_message: str,
        skill_slug: str,
        token: str,
        session_id: Optional[str],
    ) -> Any:
        """Body of the expansion; runs with the session platform contextvar
        bound to zet_agent (see caller). The fork's skill-scope resolvers
        read that ContextVar BEFORE the process HERMES_PLATFORM env, so the
        binding is authoritative here without touching global state."""
        try:
            from agent.skill_commands import (
                build_skill_invocation_message,
                scan_skill_commands,
            )
            commands = scan_skill_commands()
        except Exception:
            logger.warning(
                "[zet_agent] skill scan failed; passing message through",
                exc_info=True,
            )
            return user_message
        info = commands.get(token)
        if not info:
            logger.warning(
                "[zet_agent] requested skill %s not installed (App inventory "
                "drift?); passing message through", skill_slug,
            )
            return user_message
        try:
            from tools.skills_tool import _is_skill_disabled

            if _is_skill_disabled(
                info.get("name") or skill_slug, platform="zet_agent"
            ):
                logger.info(
                    "[zet_agent] skill %s is disabled for zet_agent; "
                    "passing message through", skill_slug,
                )
                return user_message
        except Exception:
            # _is_skill_disabled fail-opens internally; only an import
            # failure lands here — degrade to the scan-level filter.
            logger.warning(
                "[zet_agent] skill %s disabled-check failed; "
                "continuing with scan-level filter only", skill_slug,
                exc_info=True,
            )

        # Task text = the message minus the quick-pick's visible token(s).
        # The token may sit anywhere (the pick appends at the cursor) and
        # may repeat (re-selects); strip standalone occurrences only, so
        # a genuine mention like "path/to/x" is never touched.
        task_text = re.sub(
            r"(?<!\S)" + re.escape(token) + r"(?!\S)", "", user_message
        )
        task_text = "\n".join(
            line for line in (l.rstrip() for l in task_text.splitlines()) if line
        ).strip()

        try:
            # task_id = the resolved chat session, so ${HERMES_SESSION_ID}
            # templates and session-scoped skill state resolve against the
            # REAL session — CLI/gateway slash parity (review P1).
            part = build_skill_invocation_message(
                token, user_instruction=task_text, task_id=session_id or None,
            )
        except Exception:
            logger.warning(
                "[zet_agent] skill %s build failed; passing message through",
                skill_slug, exc_info=True,
            )
            return user_message
        if not part:
            logger.warning(
                "[zet_agent] skill %s resolved by scan but failed to "
                "load; passing message through", skill_slug,
            )
            return user_message
        logger.info(
            "[zet_agent] expanded skill invocation %s (task_chars=%d)",
            skill_slug, len(task_text),
        )
        return part

    async def _emit_native_session_title(
        self,
        *,
        result: Any,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        session_id: Optional[str],
        stream_q: Any,
        agent_ref: Any,
        gateway_session_key: Optional[str],
    ) -> None:
        """Run Hermes' native title worker before the request stream closes."""
        if not isinstance(result, tuple) or not result or not isinstance(result[0], dict):
            return
        run_result = result[0]
        if _chat_finish_reason_from_result(run_result) == "error":
            return
        assistant_response = str(run_result.get("final_response") or "").strip()
        effective_session_id = str(run_result.get("session_id") or session_id or "").strip()
        if not user_message or not assistant_response or not effective_session_id:
            return

        agent = agent_ref[0] if isinstance(agent_ref, list) and agent_ref else None
        session_db = getattr(agent, "_session_db", None)
        if session_db is None:
            session_db = await self._ensure_session_db_async()
        if session_db is None:
            return

        all_messages = run_result.get("messages")
        if not isinstance(all_messages, list) or not all_messages:
            all_messages = list(conversation_history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]

        def _title_failure_cb(task: str, exc: BaseException) -> None:
            logger.debug("[zet_agent] native %s failed: %s", task, exc)

        main_runtime = None
        runtime_getter = getattr(agent, "_current_main_runtime", None)
        if callable(runtime_getter):
            try:
                main_runtime = runtime_getter()
            except Exception:
                logger.debug("[zet_agent] failed to read current main runtime", exc_info=True)
        if not isinstance(main_runtime, dict) and agent is not None:
            main_runtime = {
                "model": getattr(agent, "model", None),
                "provider": getattr(agent, "provider", None),
                "base_url": getattr(agent, "base_url", None),
                "api_key": getattr(agent, "api_key", None),
                "api_mode": getattr(agent, "api_mode", None),
            }

        from agent.title_generator import maybe_auto_title

        def _run_title_worker() -> None:
            from gateway.session_context import clear_session_vars

            tokens = self._bind_api_server_session(
                chat_id=effective_session_id,
                session_key=gateway_session_key or effective_session_id,
                session_id=effective_session_id,
            )
            try:
                maybe_auto_title(
                    session_db,
                    effective_session_id,
                    user_message,
                    assistant_response,
                    all_messages,
                    failure_callback=_title_failure_cb,
                    main_runtime=main_runtime,
                    title_callback=(
                        (lambda title: self._push_title(stream_q, title))
                        if stream_q is not None
                        else None
                    ),
                    background=False,
                )
            finally:
                clear_session_vars(tokens)

        if stream_q is None:
            threading.Thread(
                target=_run_title_worker,
                daemon=True,
                name="zet-agent-auto-title",
            ).start()
            return
        await asyncio.to_thread(_run_title_worker)

    @staticmethod
    def _push_steer_dropped_if_any(stream_q: Any, run_result: Any) -> None:
        """Surface an unconsumed /steer as a ``steer_dropped`` progress event.

        ``run_result`` is base ``_run_agent``'s ``(result_dict, usage)``
        tuple; the finalizer puts leftover steer text under
        ``result_dict["pending_steer"]``. Read-only — the dict is returned
        to the caller untouched. Best-effort: a push failure must never
        fail the turn that just completed.
        """
        if stream_q is None:
            return
        try:
            result_dict = run_result[0] if isinstance(run_result, tuple) else run_result
            if not isinstance(result_dict, dict):
                return
            leftover = result_dict.get("pending_steer")
            if not leftover or not str(leftover).strip():
                return
            stream_q.put((
                "__tool_progress__",
                {"type": "steer_dropped", "text": str(leftover)},
            ))
        except Exception:
            logger.debug("[zet_agent] steer_dropped push failed", exc_info=True)

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
    # Clarify — stable instance IDs with legacy FIFO fallback on respond
    # ------------------------------------------------------------------

    def _make_clarify_cb(self, stream_q: Any, session_id: str):
        """Return a sync ``(question, choices) -> str`` callback.

        Appends an entry to the session's FIFO queue and pushes a
        ``hermes.clarify`` event onto ``_stream_q``, then blocks on
        the entry's ``threading.Event`` until the HTTP respond handler
        pops the entry and signals it. Timeout returns "" so a stale
        clarify never hangs the turn forever.

        A new opaque ``clarify_id`` is attached before either the SSE event
        or reconnect projection is published. The response endpoint uses this
        identity when the client provides it; legacy clients that do not yet
        send the field retain the historical FIFO response behavior.
        """
        # Capture profile identity while the callback is attached. The agent
        # invokes it later from its worker thread, where the request's profile
        # contextvar need not be active any more.
        scoped_session_key = self._active_turn_key(session_id)

        def _ask(question: str, choices: Optional[List[str]]) -> str:
            # Stamp the deadline using the same constant the agent
            # thread waits on a few lines below. Clients see the wall-
            # clock time we will actually give up at.
            expires_at_ms = int((time.time() + CLARIFY_RESPONSE_TIMEOUT) * 1000)
            clarify_id = uuid.uuid4().hex
            payload = {
                "type": "hermes.clarify",
                "clarify_id": clarify_id,
                "question": question,
                "choices_offered": list(choices or []),
                "expires_at_ms": expires_at_ms,
            }
            entry = _ClarifyEntry(clarify_id, payload)
            with self._clarify_state_lock:
                queue = self._clarify_queues.setdefault(scoped_session_key, [])
                queue.append(entry)
                # /pending is the projection of the entry legacy clients
                # would answer next.  Keep that projection on FIFO's head
                # even if a future producer can enqueue concurrently.
                is_pending_head = len(queue) == 1
            with self._pending_lock:
                if is_pending_head:
                    self._pending_clarify[scoped_session_key] = payload
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] clarify push failed", exc_info=True)
                self._discard_clarify_entry(scoped_session_key, entry)
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
                self._discard_clarify_entry(scoped_session_key, entry)
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
    # Delegation progress — subagent lifecycle relayed onto the SSE lane
    # ------------------------------------------------------------------

    # Child-relay events forwarded to the App. subagent.text / .thinking are
    # deliberately dropped: they stream the child's full prose (unbounded
    # volume) and the drill-down view reads it from the live transcript files
    # instead. The SSE lane carries status-level progress only.
    _DELEGATION_PROGRESS_EVENTS = frozenset(
        {"subagent.start", "subagent.tool", "subagent.progress", "subagent.complete"}
    )
    # Structured identity kwargs relayed by delegate_tool's child callback
    # (_relay → parent_cb(..., **identity_kwargs)) that the App needs to
    # address a row in the progress banner.
    _DELEGATION_PROGRESS_FIELDS = (
        "task_index",
        "task_count",
        "goal",
        "subagent_id",
        "parent_id",
        "depth",
        "child_session_id",
        "tool_count",
        "status",
        "duration_seconds",
        "exit_reason",
    )
    _DELEGATION_PREVIEW_MAX = 200
    # Cap for progress frames parked in a stream_q with no live SSE reader
    # (background children outliving the parent turn). See _cb note.
    _DELEGATION_PROGRESS_BACKLOG_MAX = 2000
    _ATTACHMENT_STREAM_BACKLOG_MAX = 2000
    _ATTACHMENT_MAX_BYTES = 256 * 1024
    _ATTACHMENT_ACTION_QUEUE_MAX = 64
    _ATTACHMENT_ACTION_WORKERS = 4
    _MEMORY_CITATION_MAX_ITEMS = 8

    @classmethod
    def _push_memory_citations(
        cls,
        stream_q: Any,
        turn_id: Any,
        session_id: Any,
        items: list,
    ) -> bool:
        """Push one turn-level memory.citations attachment (需求 3.1).

        同 turn 恒定附件 id（upsert 语义）：重复 flush 覆盖而非叠卡。无 actions、
        state 恒 active——客户端渲染为回答尾部的折叠角标行。背压/超限直接放弃
        （引用展示是旁路产物）。
        """
        try:
            if stream_q.qsize() > cls._ATTACHMENT_STREAM_BACKLOG_MAX:
                return False
            trimmed = [
                {
                    "id": str(item.get("id") or "")[:64],
                    "source": str(item.get("source") or "")[:120],
                    "excerpt": str(item.get("excerpt") or "")[:240],
                }
                for item in items[: cls._MEMORY_CITATION_MAX_ITEMS]
                if isinstance(item, dict) and item.get("id")
            ]
            if not trimmed:
                return False
            anchor = str(turn_id or "").strip() or uuid.uuid5(
                uuid.NAMESPACE_OID, f"mc:{session_id}"
            ).hex[:12]
            stream_q.put((
                "__tool_progress__",
                {
                    "type": "hermes.attachment",
                    "attachment": {
                        "id": f"mc-{anchor}",
                        "kind": "memory.citations",
                        "v": 1,
                        "state": "active",
                        "payload": {"items": trimmed},
                    },
                },
            ))
            return True
        except Exception:
            logger.warning("[zet_agent] memory citations push failed", exc_info=True)
            return False

    @classmethod
    def _push_memory_saved(
        cls,
        stream_q: Any,
        turn_id: Any,
        session_id: Any,
        items: list,
    ) -> bool:
        """Push one turn-level memory.saved attachment（记忆写入透明化）。

        与 _push_memory_citations 同款语义：同 turn 恒定 id（ms-<turn>）upsert、
        无 actions、state 恒 active，客户端渲染为「记住了 N 条」折叠角标。
        背压/超限放弃——透明化是旁路产物。"""
        try:
            if stream_q.qsize() > cls._ATTACHMENT_STREAM_BACKLOG_MAX:
                return False
            trimmed = [
                {
                    "id": str(item.get("id") or "")[:64],
                    "source": str(item.get("source") or "")[:120],
                    "excerpt": str(item.get("excerpt") or "")[:240],
                }
                for item in items[: cls._MEMORY_CITATION_MAX_ITEMS]
                if isinstance(item, dict) and item.get("id")
            ]
            if not trimmed:
                return False
            anchor = str(turn_id or "").strip() or uuid.uuid5(
                uuid.NAMESPACE_OID, f"ms:{session_id}"
            ).hex[:12]
            stream_q.put((
                "__tool_progress__",
                {
                    "type": "hermes.attachment",
                    "attachment": {
                        "id": f"ms-{anchor}",
                        "kind": "memory.saved",
                        "v": 1,
                        "state": "active",
                        "payload": {"items": trimmed},
                    },
                },
            ))
            return True
        except Exception:
            logger.warning("[zet_agent] memory saved push failed", exc_info=True)
            return False

    @classmethod
    def _make_delegation_progress_cb(cls, stream_q: Any):
        """Return a parent ``tool_progress_callback`` bridging child progress.

        delegate_task's ``_build_child_progress_callback`` relays child
        lifecycle events to ``parent_agent.tool_progress_callback`` — a
        callback the chat-completions path never wired before, so gateway
        children ran blind. This bridge forwards the status-level subset onto
        the ``hermes.tool.progress`` SSE extension lane as
        ``type=hermes.delegation.progress`` payloads (local-server translates
        them into chatproto delegation events for the App).

        Contract notes:
        - signature mirrors the relay: ``(event, tool_name, preview, args,
          **identity_kwargs)``;
        - ``subagent_progress`` (nested-orchestrator pass-through, summary in
          the tool_name slot) is normalised to ``subagent.progress``;
        - never raises into the agent loop.
        """

        def _cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
            try:
                event = str(event_type or "")
                if event == "subagent_progress":
                    event = "subagent.progress"
                    if preview is None:
                        preview = tool_name
                        tool_name = None
                if event not in cls._DELEGATION_PROGRESS_EVENTS:
                    return
                payload: Dict[str, Any] = {
                    "type": "hermes.delegation.progress",
                    "kind": "delegation",
                    "event": event,
                }
                if tool_name:
                    payload["tool"] = str(tool_name)
                if preview:
                    text = str(preview)
                    if len(text) > cls._DELEGATION_PREVIEW_MAX:
                        text = text[: cls._DELEGATION_PREVIEW_MAX] + "…"
                    payload["preview"] = text
                for field in cls._DELEGATION_PROGRESS_FIELDS:
                    value = kwargs.get(field)
                    if value is not None:
                        payload[field] = value
                # Background children capture this callback at dispatch and
                # keep pushing after the parent turn's SSE writer exits —
                # nobody drains the queue then. Cap the backlog (HR#1);
                # progress is best-effort UI signal, the live manifest is
                # the authoritative record the App polls for terminal state.
                if stream_q.qsize() > cls._DELEGATION_PROGRESS_BACKLOG_MAX:
                    return
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug(
                    "[zet_agent] delegation progress push failed", exc_info=True
                )

        return _cb

    # ------------------------------------------------------------------
    # Plan emit — non-blocking, fires when agent calls present_plan
    # ------------------------------------------------------------------

    @staticmethod
    def _make_plan_emit_cb(stream_q: Any, agent: Any):
        """Return a sync ``(title, groups) -> None`` callback.

        Called by tool_executor when the agent invokes present_plan.
        Pushes a ``hermes.plan`` event onto the SSE extension lane.
        present_plan() returns an instruction to the agent immediately after,
        so this callback never blocks.

        ``auto_execute`` on the payload tells the App whether this is a
        read-only auto-execute card (agent keeps executing in the same turn) or
        the legacy confirmation card (App gates execution on a user tap). It
        follows the resolved turn-level auto-execute flag (App capability opt-in
        > env kill-switch > default False/manual) regardless of whether the App
        requested plan mode or the model presented a plan on its own. Only a
        client that opted in (or the env kill-switch) turns it into the read-only
        auto card; every other case stays the legacy confirmation card.
        """
        def _emit(title: str, groups: List[Dict[str, Any]]) -> None:
            payload = {
                "type": "hermes.plan",
                "title": title,
                "groups": groups,
                "auto_execute": bool(getattr(agent, "_zet_agent_plan_auto_execute", False)),
            }
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] plan emit push failed", exc_info=True)

        return _emit

    def _discard_clarify_entry(self, scoped_session_key: str, entry: _ClarifyEntry) -> None:
        """Remove an unresolved entry (push failure or timeout). The
        respond handler removes via popleft on success; this path
        handles error rollback so the queue doesn't accumulate."""
        next_payload: Optional[Dict[str, Any]] = None
        with self._clarify_state_lock:
            queue = self._clarify_queues.get(scoped_session_key)
            if queue and entry in queue:
                queue.remove(entry)
            if queue is not None and not queue:
                self._clarify_queues.pop(scoped_session_key, None)
            elif queue:
                # The pending projection always represents the entry at the
                # front of the legacy FIFO. This also keeps reconnect correct
                # should a future producer create more than one entry.
                next_payload = queue[0].payload
        with self._pending_lock:
            current = self._pending_clarify.get(scoped_session_key)
            if current and current.get("clarify_id") == entry.clarify_id:
                if next_payload is None:
                    self._pending_clarify.pop(scoped_session_key, None)
                else:
                    self._pending_clarify[scoped_session_key] = next_payload

    def _register_active_session_turn(self, session_id: Optional[str], agent_ref: list, agent_task: Any) -> None:
        """Stash the in-flight chat-completions turn so the session
        interrupt endpoint can reach it. agent_ref is the mutable
        ``[None] -> [AIAgent]`` list the base handler fills in once the
        agent is constructed; we keep the list itself (not a snapshot)
        so the interrupt picks up the agent the moment it appears."""
        if not session_id:
            return
        # Key by profile home + sid (codex P1): under the multiplexer two
        # profiles can run same-named sessions CONCURRENTLY — a bare-sid key
        # would let the later registration overwrite the earlier one, whose
        # goal driver then can't see its own in-flight turn and double-drives
        # the loop (reconcile/resume/barrier wakeup). The contextvar scope is
        # live here (the chat request entered via /p/{profile}).
        key = self._active_turn_key(session_id)
        with self._session_run_lock:
            self._active_session_agents[key] = agent_ref
            self._active_session_tasks[key] = agent_task

    def _active_turn_key(self, session_id: str) -> str:
        """Scoped registry key: {hermes_home}|{session_id} — same shape as
        the goal driver's _scope_key so both sides resolve identically."""
        try:
            from hermes_constants import get_hermes_home

            return f"{get_hermes_home()}|{session_id}"
        except Exception:
            return session_id

    def _clear_active_session_turn(self, session_id: Optional[str], agent_ref: list, agent_task: Any) -> None:
        """Drop the registration ONLY if it still points at the turn we
        registered. Guards against late-clearing a fresher turn that
        the same session has already started. Identity-scan fallback: if the
        clear runs outside the registration's profile scope the scoped key
        won't reconstruct — a leaked entry would read as a forever-live turn,
        so hunt the exact agent_ref down."""
        if not session_id:
            return
        key = self._active_turn_key(session_id)
        with self._session_run_lock:
            if self._active_session_agents.get(key) is agent_ref:
                self._active_session_agents.pop(key, None)
            else:
                for k, v in list(self._active_session_agents.items()):
                    if v is agent_ref:
                        self._active_session_agents.pop(k, None)
                        break
            if self._active_session_tasks.get(key) is agent_task:
                self._active_session_tasks.pop(key, None)
            else:
                for k, v in list(self._active_session_tasks.items()):
                    if v is agent_task:
                        self._active_session_tasks.pop(k, None)
                        break

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
        route: Optional[Dict[str, Any]] = None,
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
        # Pull the zet_agent-only Plan auto-execute hint out of request_overrides
        # before it can reach the AIAgent (and the LLM request body). _run_agent
        # stamps the resolved per-turn flag here so the Plan-First addendum section
        # matches the turn's confirm/auto behaviour. Absent (async /v1/runs path,
        # or non-plan callers) → None → manual (safe default).
        agent_request_overrides = dict(request_overrides or {})
        plan_auto_execute = agent_request_overrides.pop(
            "_zet_plan_auto_execute", None
        )
        disable_tools = agent_request_overrides.pop("tool_choice", None) == "none"

        # 在 ephemeral_system_prompt 头部接 zettlab 工作风格 addendum。
        # 上游传进来的 ephemeral 通常是 SOUL.md / IDENTITY.md 的拼接（per-agent
        # 人格），让 addendum 在前、SOUL 在后是有意的：模型在系统提示里靠后
        # 的 instruction 优先级更高，per-agent SOUL 真要 override 这条 workflow
        # 时仍能压过去。
        ephemeral_system_prompt = (
            _zettlab_workflow_addendum(bool(plan_auto_execute))
            + ("\n\n" + ephemeral_system_prompt if ephemeral_system_prompt else "")
        )

        # cron origin + async-delivery capability（见 helper docstring）。
        self._bind_turn_session_context(session_id)

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

        # Keep parity with APIServerAdapter: a fallback runtime may carry its
        # own model, and passing it alongside the explicit model argument would
        # otherwise raise "multiple values for keyword argument 'model'".
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            model = runtime_model

        # ZET-576: apply session-level model override if present.
        # _resolve_gateway_model reads config.yaml (agent default), but
        # session overrides live in gateway_runner._session_model_overrides
        # which this adapter's _create_agent bypasses. Check it here.
        gw = getattr(self, "gateway_runner", None)
        override_key = gateway_session_key or session_id
        runtime_auxiliary_task_configs = None
        runtime_supports_vision = None
        override = None
        if gw is not None and override_key:
            candidate = getattr(gw, "_session_model_overrides", {}).get(override_key)
            if isinstance(candidate, dict):
                override = dict(candidate)
        if override is None:
            override = self._session_model_override_for(override_key)

        # Per-client route sits between the global runtime and an explicit
        # session /model override, matching the base API adapter precedence.
        if route and not override:
            if route.get("provider"):
                try:
                    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

                    provider_kwargs = _resolve_runtime_agent_kwargs_for_provider(
                        route["provider"]
                    )
                    provider_kwargs.pop("model", None)
                    runtime_kwargs.update(provider_kwargs)
                except Exception:
                    for key in (
                        "api_key",
                        "base_url",
                        "api_mode",
                        "command",
                        "args",
                        "credential_pool",
                    ):
                        runtime_kwargs.pop(key, None)
                    runtime_kwargs["provider"] = route["provider"]
            if route.get("model"):
                model = route["model"]
            if route.get("api_key"):
                runtime_kwargs["api_key"] = route["api_key"]
            if route.get("base_url"):
                runtime_kwargs["base_url"] = route["base_url"]
            logger.debug(
                "zet_agent model route applied: model=%s provider=%s",
                model,
                runtime_kwargs.get("provider"),
            )
        elif route and override:
            logger.debug(
                "zet_agent model route skipped: session /model override wins for %s",
                override_key,
            )

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
            request_overrides=agent_request_overrides or None,
        )
        if disable_tools:
            agent.tools = []
            agent.valid_tool_names = set()
            agent._skip_mcp_refresh = True
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
            agent.plan_emit_callback = self._make_plan_emit_cb(stream_q, agent)
        except Exception:
            logger.warning("[zet_agent] failed to attach plan_emit_callback", exc_info=True)

        # 3d. Delegation progress: bridge delegate_task child lifecycle
        # events onto the SSE lane (type=hermes.delegation.progress).
        # Wiring the parent tool_progress_callback is also what ENABLES
        # delegate_tool's child relay on this path (it returns no callback
        # when the parent has neither spinner nor progress callback).
        # Rebind UNCONDITIONALLY each turn (mirrors plan_emit_callback):
        # session agents are reused across turns, and a keep-if-set guard
        # would leave the callback closed over the FIRST turn's dead
        # stream_q — every later dispatch's progress would go to a queue
        # nobody drains (invisible + unbounded backlog).
        try:
            agent.tool_progress_callback = self._make_delegation_progress_cb(
                stream_q
            )
        except Exception:
            logger.warning(
                "[zet_agent] failed to attach delegation progress callback",
                exc_info=True,
            )

        # 4. Approval: register a per-session notify callback.
        # We don't unregister here because chat.completions reuses the
        # same session_id across turns; unregistration happens on
        # platform disconnect (or never, for short-lived processes).
        if session_id:
            try:
                from tools.approval import register_gateway_notify
                register_gateway_notify(session_id, self._make_approval_cb(stream_q, session_id))
                with self._session_lock:
                    self._approval_session_ids.add(session_id)
            except Exception:
                logger.warning("[zet_agent] failed to register approval notify", exc_info=True)

        # 5. Auto-title runs after the first successful assistant reply.

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
        route: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        plan_ack: Optional[Dict[str, Any]] = None,
        plan_auto_execute: Optional[bool] = None,
        turn_id: Optional[str] = None,
        connector_route_capability: Optional[str] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Wrap base ``_run_agent`` to bind the session-scoped env
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
        title_user_message = self._title_user_message(user_message)

        # Open-time check: if this session's effective model (override, else
        # config default) differs from the persisted last-seen value, inject a
        # one-shot identity note. Covers session- and agent-level switches,
        # survives restarts; a brand-new session just records its baseline.
        attachment_emitter_token = None
        if stream_q is not None:
            try:
                from hermes_cli.plugins import bind_attachment_emitter

                def _emit_attachment(attachment: Dict[str, Any]) -> bool:
                    try:
                        if not isinstance(attachment, dict):
                            return False
                        encoded = json.dumps(
                            attachment,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        if len(encoded) > self._ATTACHMENT_MAX_BYTES:
                            logger.warning(
                                "[zet_agent] attachment payload rejected: %d bytes",
                                len(encoded),
                            )
                            return False
                        if stream_q.qsize() > self._ATTACHMENT_STREAM_BACKLOG_MAX:
                            logger.warning(
                                "[zet_agent] attachment stream backlog saturated"
                            )
                            return False
                        # Round-trip JSON to detach the queued frame from plugin
                        # mutation after emit_attachment returns.
                        safe_attachment = json.loads(encoded)
                        stream_q.put((
                            "__tool_progress__",
                            {
                                "type": "hermes.attachment",
                                "attachment": safe_attachment,
                            },
                        ))
                        return True
                    except Exception:
                        logger.warning(
                            "[zet_agent] attachment stream push failed",
                            exc_info=True,
                        )
                        return False

                attachment_emitter_token = bind_attachment_emitter(
                    _emit_attachment
                )
            except Exception:
                logger.warning(
                    "[zet_agent] failed to bind attachment emitter",
                    exc_info=True,
                )

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
                route=route,
                response_mode=response_mode,
                plan_ack=plan_ack,
                plan_auto_execute=plan_auto_execute,
                turn_id=turn_id,
                connector_route_capability=connector_route_capability,
                request_overrides=request_overrides,
            )
            # Early-return steer salvage: many conversation_loop retry/error
            # paths return without running finalize_turn, so the closing
            # drain never happens — a steer accepted in those windows would
            # have no consumer and no dropped receipt (silently lost with the
            # turn). If the result carries no pending_steer but the slot
            # still holds text, drain it here (close=True so later steers
            # are refused) and let the receipt push below re-queue it. On
            # the normal finalize path this is a no-op (slot already drained
            # and closed).
            try:
                _salvage_agent = agent_ref[0] if agent_ref else None
                if (
                    _salvage_agent is not None
                    and isinstance(result, tuple)
                    and result
                    and isinstance(result[0], dict)
                    and not result[0].get("pending_steer")
                ):
                    _leftover = _salvage_agent._drain_pending_steer(close=True)
                    if _leftover:
                        result[0]["pending_steer"] = _leftover
                        logger.info(
                            "[zet_agent] salvaged steer from early-return turn session=%s",
                            session_id,
                        )
            except Exception:
                logger.debug("[zet_agent] early-return steer salvage failed", exc_info=True)
            # memory.citations（需求 3.1）：本轮 search_memory 命中的记忆条目
            # 汇总成一张附件（agent 侧 dispatch 采集到 _zet_memory_citations）。
            # 仍在 agent_task 内（None 哨兵未落），push 一定会被 drain。采集/
            # 发射失败一律静默——引用展示是旁路，绝不影响回答。
            try:
                _cit_agent = agent_ref[0] if agent_ref else None
                if _cit_agent is not None and not getattr(_cit_agent, "_zet_memory_citations", None):
                    # 常驻系统提示的记忆没有采集点（模型不调工具就无引用）——
                    # 用回答文本对记忆条目做事后归因（相对阈值压误报）。
                    _final_text = ""
                    if isinstance(result, tuple) and result and isinstance(result[0], dict):
                        _final_text = str(result[0].get("final_response") or "")
                    from agent.agent_runtime_helpers import collect_answer_attribution_citations
                    collect_answer_attribution_citations(_cit_agent, _final_text)
                citations = getattr(_cit_agent, "_zet_memory_citations", None)
                if stream_q is not None and isinstance(citations, dict) and citations:
                    self._push_memory_citations(
                        stream_q, turn_id, session_id, list(citations.values())
                    )
                    _cit_agent._zet_memory_citations = {}
                saves = getattr(_cit_agent, "_zet_memory_saves", None)
                if stream_q is not None and isinstance(saves, dict) and saves:
                    self._push_memory_saved(
                        stream_q, turn_id, session_id, list(saves.values())
                    )
                    _cit_agent._zet_memory_saves = {}
            except Exception:
                logger.debug("[zet_agent] memory citations push failed", exc_info=True)
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
                # Unconsumed /steer wins over goal continuation: the text is
                # about to be surfaced as steer_dropped below and re-queued by
                # the client as the next turn. Scheduling the judge now races
                # its continuation kick against the user's redirect — the
                # autopilot could out-run the correction. Skip this round; the
                # re-queued message's own post-turn hook re-enters the loop.
                has_pending_steer = False
                if isinstance(result, tuple) and result and isinstance(result[0], dict):
                    has_pending_steer = bool(result[0].get("pending_steer"))
                if run_ok and not has_pending_steer:
                    # Consumed mid-turn steer = the user intervened in this
                    # round. The goal judge keys user_initiated off the
                    # message NOT starting with CONTINUATION_MARKER — pass
                    # the steer text so an auto-continuation round the user
                    # redirected is evaluated as user-initiated instead of
                    # the autopilot overriding the correction.
                    _consumed_steer = None
                    try:
                        _agent_for_steer = agent_ref[0] if agent_ref else None
                        _consumed_steer = getattr(_agent_for_steer, "_turn_last_steer_text", None)
                    except Exception:
                        _consumed_steer = None
                    self._goals().schedule_after_turn(
                        session_id or "",
                        _consumed_steer or user_message,
                        final_response,
                        effective_session_id=effective_sid,
                    )
                elif run_ok:
                    logger.info(
                        "[zet_agent] goal post-turn hook deferred: unconsumed steer pending session=%s",
                        session_id,
                    )
                else:
                    logger.info(
                        "[zet_agent] goal post-turn hook skipped for failed turn session=%s",
                        session_id,
                    )
            except Exception:
                logger.debug("[zet_agent] goal post-turn hook failed", exc_info=True)
            # A /steer that landed after the final tool boundary was drained
            # into result["pending_steer"] by run_conversation's finalizer.
            # CLI/gateway re-deliver it as the next user turn; the stateless
            # chat-completions path instead surfaces a steer_dropped progress
            # event so local-server/App can re-queue the text. This MUST run
            # here — before this coroutine returns — because the SSE writer's
            # close sentinel is enqueued by agent_task's done callback, and
            # anything put on stream_q after that may never be drained.
            self._push_steer_dropped_if_any(stream_q, result)
            if not title_user_message.startswith("[ZETTLAB:"):
                try:
                    await self._emit_native_session_title(
                        result=result,
                        user_message=title_user_message,
                        conversation_history=conversation_history,
                        session_id=session_id,
                        stream_q=stream_q,
                        agent_ref=agent_ref,
                        gateway_session_key=gateway_session_key,
                    )
                except Exception:
                    logger.debug("[zet_agent] native auto-title hook failed", exc_info=True)
            return result
        finally:
            if attachment_emitter_token is not None:
                try:
                    from hermes_cli.plugins import reset_attachment_emitter
                    reset_attachment_emitter(attachment_emitter_token)
                except Exception:
                    logger.warning(
                        "[zet_agent] failed to reset attachment emitter",
                        exc_info=True,
                    )
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
        """POST /v1/sessions/{session_id}/clarify/respond — answer one
        pending clarify prompt.

        Body: ``{"response": "...", "clarify_id": "..."}``. A supplied
        id must match a live entry and is never allowed to consume another
        prompt. Omitting it preserves the legacy oldest-pending FIFO behavior.
        404 makes stale or mismatched retries explicit rather than silently
        applying an answer to a different card.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info.get("session_id", "")
        scoped_session_key = self._active_turn_key(session_id)
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        response_text = str(body.get("response", "") or "")
        clarify_id = str(body.get("clarify_id", "") or "").strip()

        next_payload: Optional[Dict[str, Any]] = None
        with self._clarify_state_lock:
            queue = self._clarify_queues.get(scoped_session_key)
            entry: Optional[_ClarifyEntry] = None
            if queue:
                if clarify_id:
                    for i, candidate in enumerate(queue):
                        if candidate.clarify_id == clarify_id:
                            entry = queue.pop(i)
                            break
                else:
                    entry = queue.pop(0)
            if queue is not None and not queue:
                self._clarify_queues.pop(scoped_session_key, None)
            elif queue:
                next_payload = queue[0].payload
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
            current = self._pending_clarify.get(scoped_session_key)
            if current and current.get("clarify_id") == entry.clarify_id:
                if next_payload is None:
                    self._pending_clarify.pop(scoped_session_key, None)
                else:
                    self._pending_clarify[scoped_session_key] = next_payload
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
        scoped_session_key = self._active_turn_key(session_id)
        with self._pending_lock:
            ap = self._pending_approval.get(session_id)
            cl = self._pending_clarify.get(scoped_session_key)
        return web.json_response({
            "approval": ap,
            "clarify": cl,
        })

    async def _handle_attachment_action(self, request: "web.Request") -> "web.Response":
        """Queue a validated attachment action for plugin observers."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = str(request.match_info.get("session_id", "") or "").strip()
        if not session_id or len(session_id.encode("utf-8")) > 256:
            return web.json_response(
                _openai_error("Invalid session ID", code="invalid_session_id"),
                status=400,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if not isinstance(body, dict):
            return web.json_response(_openai_error("Body must be an object"), status=400)

        limits = {
            "attachment_id": 256,
            "action_id": 128,
            "action_token": 256,
            "turn_id": 256,
        }
        normalized: Dict[str, str] = {}
        for field, limit in limits.items():
            value = body.get(field, "")
            if value is None and field == "turn_id":
                value = ""
            if not isinstance(value, str):
                return web.json_response(
                    _openai_error(f"{field} must be a string"), status=400
                )
            value = value.strip()
            if field != "turn_id" and not value:
                return web.json_response(
                    _openai_error(f"{field} is required"), status=400
                )
            if len(value.encode("utf-8")) > limit:
                return web.json_response(
                    _openai_error(f"{field} is too long"), status=400
                )
            normalized[field] = value

        payload = body.get("payload")
        if payload is not None and not isinstance(payload, dict):
            return web.json_response(
                _openai_error("payload must be an object"), status=400
            )
        if payload is not None:
            payload_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            if payload_size > self._ATTACHMENT_MAX_BYTES:
                return web.json_response(
                    _openai_error("payload is too large"), status=413
                )

        profile_name = str(_request_value(request, "hermes_profile") or "default")
        hook_data: Dict[str, Any] = {
            "session_id": session_id,
            "attachment_id": normalized["attachment_id"],
            "action_id": normalized["action_id"],
            "action_token": normalized["action_token"],
            "turn_id": normalized["turn_id"],
            "payload": payload,
            "profile_name": profile_name,
        }
        queue = self._ensure_attachment_action_workers()
        try:
            queue.put_nowait((profile_name, hook_data))
        except asyncio.QueueFull:
            return web.json_response(
                _openai_error(
                    "attachment action queue is saturated",
                    code="attachment_action_saturated",
                ),
                status=503,
            )
        return web.json_response({"accepted": True}, status=202)

    def _ensure_attachment_action_workers(self) -> "asyncio.Queue":
        queue = getattr(self, "_attachment_action_queue", None)
        if queue is not None:
            return queue
        queue = asyncio.Queue(maxsize=self._ATTACHMENT_ACTION_QUEUE_MAX)
        self._attachment_action_queue = queue
        for _ in range(self._ATTACHMENT_ACTION_WORKERS):
            task = asyncio.create_task(self._attachment_action_worker(queue))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        return queue

    async def _attachment_action_worker(self, queue: "asyncio.Queue") -> None:
        while True:
            profile_name, hook_data = await queue.get()
            try:
                def _invoke() -> None:
                    from hermes_cli.plugins import invoke_hook
                    if profile_name and profile_name != "default":
                        with self._profile_api_scope(profile_name):
                            invoke_hook("attachment_action", **hook_data)
                    else:
                        invoke_hook("attachment_action", **hook_data)

                await asyncio.to_thread(_invoke)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "[zet_agent] attachment_action hook dispatch failed",
                    exc_info=True,
                )
            finally:
                queue.task_done()

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
        # Scoped key first（注册键含 profile home），裸键回退兼容 legacy 注册。
        turn_key = self._active_turn_key(session_id)
        with self._session_run_lock:
            agent_ref = self._active_session_agents.get(turn_key) or self._active_session_agents.get(session_id)
            task = self._active_session_tasks.get(turn_key) or self._active_session_tasks.get(session_id)

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

        # Also stop this session's in-flight background delegations.
        # ``agent.interrupt()`` cannot reach them: delegate_task(background)
        # deliberately detaches children from _active_children at dispatch
        # (their lifecycle is owned by the async registry). Without this,
        # "stop" looks honored in the App while subagents keep burning
        # tokens in the background. parent_session_id is the selector the
        # dispatch records (delegate_tool captures parent_agent.session_id);
        # include the agent's current session_id too in case compaction
        # rotated it since dispatch.
        try:
            from tools.async_delegation import interrupt_for_session

            rotated_sid = str(getattr(agent, "session_id", "") or "") if agent else ""
            for psid in {session_id, rotated_sid} - {""}:
                # suppress_completion: the user explicitly stopped this turn —
                # local-server anchors the batch outcome card onto the
                # interrupted turn itself, so the killed children must NOT
                # re-enter the chat with a completion turn afterwards.
                # profile scope: under a multiplexer this route must not be
                # able to kill (and suppress-swallow) ANOTHER profile's batch
                # by quoting its session id (same rule as the control plane).
                interrupt_for_session(
                    parent_session_id=psid,
                    reason="user_cancel",
                    suppress_completion=True,
                    profile_home=self._delegation_control_scope(request),
                )
        except Exception:
            logger.debug(
                "[zet_agent] session interrupt: async delegation interrupt failed",
                exc_info=True,
            )

        self._interrupt_pending_interactions(session_id, turn_key)

        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:
                logger.debug("[zet_agent] session interrupt: task.cancel failed", exc_info=True)

        status = "stopping" if (agent is not None or task is not None) else "not_running"
        return web.json_response({"session_id": session_id, "status": status})

    # ------------------------------------------------------------------
    # Delegation control plane (App banner: status / per-id cancel)
    # ------------------------------------------------------------------

    @staticmethod
    def _delegation_control_scope(request: "web.Request") -> str:
        """Resolve the profile filter for a delegation control-plane call.

        /p/{profile} routes carry ``hermes_profile_home`` (stamped by
        _profile_handler). The bare /v1 route has no stamp: under an ACTIVE
        multiplexer it is scoped to the DEFAULT profile (the unscoped
        ``get_hermes_home()``) so it can never read or cancel another
        profile's work; single-profile processes return "" (legacy full
        view — every record carries the same home anyway).
        """
        stamped = str(request.get("hermes_profile_home", "") or "")
        if stamped:
            return stamped
        try:
            from agent.secret_scope import is_multiplex_active

            if is_multiplex_active():
                from hermes_constants import get_hermes_home

                return str(get_hermes_home())
        except Exception:
            logger.debug(
                "[zet_agent] delegation control scope resolution failed",
                exc_info=True,
            )
        return ""

    async def _handle_delegations_status(self, request: "web.Request") -> "web.Response":
        """GET /v1/delegations/status — sync tree + async records snapshot.

        Thin wrapper over the same module-level registries the TUI /agents
        overlay reads (tui_gateway delegation.status). Consumed by
        zettlab-local-server for reconcile-after-restart and the App's
        control-plane proxy.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        # The registries are process-global; under a multiplexer the
        # /p/{profile} route must only ever see (and cancel) ITS OWN
        # records. The BARE /v1 route carries no profile stamp — under an
        # active multiplexer it must fall back to the DEFAULT profile's
        # scope (unscoped get_hermes_home()), not to "no filter": an empty
        # filter would leak every profile's goals/session keys through the
        # default route and let their ids be cancelled cross-profile.
        # Single-profile processes keep the legacy full view.
        profile_home = self._delegation_control_scope(request)
        try:
            from tools.async_delegation import list_async_delegations
            from tools.delegate_tool import list_active_subagents

            active = list_active_subagents(profile_home)
            async_records = list_async_delegations(profile_home)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"active": active, "async": async_records})

    async def _handle_delegation_cancel(self, request: "web.Request") -> "web.Response":
        """POST /v1/delegations/{delegation_id}/cancel — stop ONE async batch."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        delegation_id = str(request.match_info.get("delegation_id", "")).strip()
        try:
            from tools.async_delegation import interrupt_delegation

            ok = interrupt_delegation(
                delegation_id,
                profile_home=self._delegation_control_scope(request),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response(
            {"delegation_id": delegation_id, "interrupted": bool(ok)},
            status=200 if ok else 404,
        )

    async def _handle_subagent_interrupt(self, request: "web.Request") -> "web.Response":
        """POST /v1/subagents/{subagent_id}/interrupt — stop ONE sync child."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        subagent_id = str(request.match_info.get("subagent_id", "")).strip()
        try:
            from tools.delegate_tool import interrupt_subagent

            ok = interrupt_subagent(
                subagent_id,
                profile_home=self._delegation_control_scope(request),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response(
            {"subagent_id": subagent_id, "interrupted": bool(ok)},
            status=200 if ok else 404,
        )

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """Extend the base capability surface with zet_agent-only endpoints.

        local-server does NOT gate chat.steer on this (it always advertises
        capabilities.steer=true on the WS and degrades via the 404 →
        steer_dropped path against an old hermes), but the endpoint contract
        is that /v1/capabilities lists platform/API availability for external
        orchestrators. When a profile does not exist yet, target-filesystem
        publish support is finalized before the first import state mutation.
        """
        resp = await super()._handle_capabilities(request)
        if getattr(resp, "status", 200) != 200:
            return resp
        try:
            payload = json.loads(resp.body)
        except Exception:
            return resp
        from tools.memory_tool import portable_memory_import_supported

        operation_key = self._begin_runtime_import_operation(
            _request_value(request, "hermes_profile_home")
        )
        if operation_key is None:
            memory_import_supported = False
        else:
            try:
                memory_import_supported = await _to_thread_with_completion_barrier(
                    portable_memory_import_supported
                )
            finally:
                self._end_runtime_import_operation(operation_key)
        payload.setdefault("features", {})["session_steer"] = True
        payload["features"]["attachment_actions"] = True
        payload["features"]["completed_transcript_import"] = True
        payload["features"]["curated_memory_import"] = memory_import_supported
        payload.setdefault("endpoints", {})["session_steer"] = {
            "method": "POST",
            "path": "/v1/sessions/{session_id}/steer",
        }
        payload["endpoints"]["attachment_action"] = {
            "method": "POST",
            "path": "/v1/sessions/{session_id}/attachment/action",
        }
        payload["endpoints"]["completed_transcript_import"] = {
            "method": "POST",
            "path": "/api/sessions/import",
            "operations": ["stage", "commit", "abort"],
        }
        payload["endpoints"]["curated_memory_import"] = {
            "method": "POST", "path": "/api/memory/import",
            "enabled": memory_import_supported,
        }
        return web.json_response(payload)

    async def _handle_session_import(self, request: "web.Request") -> "web.Response":
        """Stage and atomically publish completed external transcripts.

        This endpoint intentionally accepts only completed user/assistant text.
        It never restores system prompts, tool calls, approvals, credentials, or
        any other in-flight runtime state.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        operation_key = self._begin_runtime_import_operation(
            _request_value(request, "hermes_profile_home")
        )
        if operation_key is None:
            return web.json_response(
                {"error": {"message": "profile is unloaded", "type": "invalid_request_error",
                           "code": "runtime_import_profile_unloaded"}},
                status=409,
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            operation = body.get("operation")
            import_id = body.get("import_id")
            # SessionDB construction opens SQLite, initializes schema and may
            # run bounded stale-import cleanup. Keep all of that off the shared
            # aiohttp loop, and retain the profile operation barrier even if
            # the request is cancelled while the worker is still opening.
            session_db = await _to_thread_with_completion_barrier(
                self._ensure_session_db
            )
            if session_db is None:
                raise RuntimeError("session db unavailable")
            if operation == "stage":
                result = await _to_thread_with_completion_barrier(
                    session_db.stage_completed_transcript_import,
                    import_id=import_id,
                    source=body.get("source"),
                    source_session_id=body.get("source_session_id"),
                    target_session_id=body.get("target_session_id"),
                    title=body.get("title"),
                    payload_sha256=body.get("payload_sha256"),
                    expected_message_count=body.get("expected_message_count"),
                    chunk_index=body.get("chunk_index"),
                    messages=body.get("messages"),
                )
            elif operation == "commit":
                result = await _to_thread_with_completion_barrier(
                    session_db.commit_completed_transcript_import, import_id
                )
            elif operation == "abort":
                result = await _to_thread_with_completion_barrier(
                    session_db.abort_completed_transcript_import, import_id
                )
            else:
                raise ValueError("operation must be stage, commit, or abort")
            return web.json_response(result)
        except Exception as exc:
            from hermes_state import RuntimeImportConflict, RuntimeImportIncomplete
            if isinstance(exc, RuntimeImportConflict):
                status, code = 409, "runtime_import_conflict"
            elif isinstance(exc, RuntimeImportIncomplete):
                status, code = 409, "runtime_import_incomplete"
            elif isinstance(exc, (ValueError, TypeError, RecursionError)):
                status, code = 400, "invalid_runtime_import"
            else:
                logger.exception("[zet_agent] completed transcript import failed")
                status, code = 500, "runtime_import_failed"
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error", "code": code}},
                status=status,
            )
        finally:
            self._end_runtime_import_operation(operation_key)

    async def _handle_memory_import(self, request: "web.Request") -> "web.Response":
        """Replace one bounded curated-memory file; effective next session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        from tools.memory_tool import portable_memory_import_supported

        operation_key = self._begin_runtime_import_operation(
            _request_value(request, "hermes_profile_home")
        )
        if operation_key is None:
            return web.json_response(
                {"error": {"message": "profile is unloaded", "type": "invalid_request_error",
                           "code": "memory_import_profile_unloaded"}},
                status=409,
            )
        try:
            if not await _to_thread_with_completion_barrier(
                portable_memory_import_supported
            ):
                return web.json_response(
                    {"error": {
                        "message": "curated memory import is unavailable on this platform",
                        "type": "invalid_request_error",
                        "code": "memory_import_unsupported",
                    }},
                    status=501,
                )
            body = await request.json()
            if not isinstance(body, dict) or body.get("mode") != "replace":
                raise ValueError("mode must be replace")
            from tools.memory_tool import load_on_disk_store

            def _import_memory():
                return load_on_disk_store(bounded=True).import_replace(
                    target=body.get("target"), entries=body.get("entries"),
                    import_id=body.get("import_id"),
                    payload_sha256=body.get("payload_sha256"),
                )

            result = await _to_thread_with_completion_barrier(
                _import_memory,
            )
            return web.json_response(result)
        except Exception as exc:
            from tools.memory_tool import (
                MemoryImportConflict,
                MemoryImportUnsupported,
            )
            if isinstance(exc, MemoryImportUnsupported):
                status, code = 501, "memory_import_unsupported"
            elif isinstance(exc, MemoryImportConflict):
                status, code = 409, "memory_import_conflict"
            elif isinstance(exc, (ValueError, TypeError, RecursionError)):
                status, code = 400, "invalid_memory_import"
            else:
                logger.exception("[zet_agent] curated memory import failed")
                status, code = 500, "memory_import_failed"
            return web.json_response(
                {"error": {"message": str(exc), "type": "invalid_request_error", "code": code}},
                status=status,
            )
        finally:
            self._end_runtime_import_operation(operation_key)

    async def _cleanup_stale_runtime_imports_once(self) -> int:
        default_key = self._profile_home_key()
        cached_candidates = [(default_key, None)]
        cached_candidates.extend(tuple(self._session_dbs.items()))
        cached_keys = {key for key, _db in cached_candidates}
        seen = set()
        deleted = 0

        for key, cached_db in cached_candidates:
            operation_key = self._begin_runtime_import_operation(key)
            if operation_key is None:
                continue
            try:
                session_db = cached_db
                if key == default_key and session_db is None:
                    session_db = await _to_thread_with_completion_barrier(
                        self._ensure_session_db
                    )
                if session_db is None or id(session_db) in seen:
                    continue
                seen.add(id(session_db))
                try:
                    deleted += await _to_thread_with_completion_barrier(
                        session_db.cleanup_stale_runtime_imports
                    )
                except Exception:
                    logger.warning(
                        "[zet_agent] runtime import staging cleanup failed for one profile",
                        exc_info=True,
                    )
            finally:
                self._end_runtime_import_operation(operation_key)

        # After a gateway restart the per-profile DB cache is empty. Discover
        # every served profile with an existing state.db so expired private
        # transcript staging still converges without a foreground request.
        try:
            profile_homes = tuple(self._multiplex_profile_homes().values())
        except Exception:
            logger.warning(
                "[zet_agent] runtime import profile discovery failed",
                exc_info=True,
            )
            profile_homes = ()
        for profile_home in profile_homes:
            key = self._profile_home_key(profile_home)
            if key in cached_keys:
                continue
            operation_key = self._begin_runtime_import_operation(key)
            if operation_key is None:
                continue
            session_db = None
            generation = self._profile_directory_identity(key)
            try:
                if generation is None:
                    continue
                session_db = await _to_thread_with_completion_barrier(
                    self._open_profile_session_db,
                    Path(profile_home),
                    create=False,
                )
                if self._profile_directory_identity(key) != generation:
                    logger.warning(
                        "[zet_agent] runtime import profile changed while DB opened; skipping cleanup"
                    )
                    continue
                deleted += await _to_thread_with_completion_barrier(
                    session_db.cleanup_stale_runtime_imports
                )
            except Exception:
                logger.warning(
                    "[zet_agent] runtime import staging cleanup failed for one profile",
                    exc_info=True,
                )
            finally:
                cancelled = None
                if session_db is not None:
                    try:
                        await _to_thread_with_completion_barrier(session_db.close)
                    except asyncio.CancelledError as exc:
                        # The helper has already waited for close to finish.
                        # Release the operation barrier before propagating the
                        # request cancellation.
                        cancelled = exc
                    except Exception:
                        logger.warning(
                            "[zet_agent] runtime import cleanup DB close failed",
                            exc_info=True,
                        )
                self._end_runtime_import_operation(operation_key)
                if cancelled is not None:
                    raise cancelled
        return deleted

    async def _sweep_stale_runtime_imports(self) -> None:
        while True:
            try:
                deleted = await self._cleanup_stale_runtime_imports_once()
                if deleted:
                    logger.info(
                        "[zet_agent] removed %d expired runtime import staging row(s)",
                        deleted,
                    )
                await asyncio.sleep(RUNTIME_IMPORT_CLEANUP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "[zet_agent] runtime import staging cleanup failed",
                    exc_info=True,
                )
                await asyncio.sleep(RUNTIME_IMPORT_CLEANUP_INTERVAL_SECONDS)

    def _register_profile_api_routes(self, router, *, chat_handler=None) -> None:
        super()._register_profile_api_routes(router, chat_handler=chat_handler)
        router.add_post(
            "/p/{profile}/api/sessions/import",
            self._profile_handler(self._handle_session_import),
        )
        router.add_post(
            "/p/{profile}/api/memory/import",
            self._profile_handler(self._handle_memory_import),
        )

    async def _handle_session_steer(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/steer — inject user text into the
        active chat-completions turn WITHOUT interrupting it.

        Companion to ``_handle_session_interrupt``: same registry lookup,
        but instead of stopping the agent it calls ``AIAgent.steer(text)``,
        which stashes the text for the conversation loop's pre-API drain
        (agent/conversation_loop.py) so the model sees it appended to the
        latest tool result on its next iteration. A steer that is never
        consumed (turn ends on a plain text response with no further tool
        boundary) is surfaced as a ``steer_dropped`` progress event by
        ``_run_agent`` so the caller can re-deliver the text as a normal
        message instead of it being silently lost.

        Body: ``{"text": "..."}`` — required, non-empty after strip.

        Returns 200 ``{accepted: true, status: "steering"}`` when the text
        was stashed onto a live agent; ``{accepted: false, status:
        "not_running"}`` when no turn is in flight for the session (caller
        should fall back to queueing the text as the next turn).
        ``agent.steer`` only takes a short self-owned lock, so calling it
        synchronously from the event loop is safe (same pattern as
        ``agent.interrupt`` above).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return web.json_response(
                _openai_error("steer requires a non-empty 'text' field"),
                status=400,
            )

        session_id = request.match_info.get("session_id", "")
        # Scoped-key lookup mirrors _handle_session_interrupt: registrations
        # are keyed by _active_turn_key ({hermes_home}|{sid}, codex P1 for
        # concurrent same-named sessions under the multiplexer); the bare-sid
        # fallback covers entries registered outside a profile scope.
        turn_key = self._active_turn_key(session_id)
        with self._session_run_lock:
            agent_ref = self._active_session_agents.get(turn_key) or self._active_session_agents.get(session_id)
            task = self._active_session_tasks.get(turn_key) or self._active_session_tasks.get(session_id)
        agent = agent_ref[0] if agent_ref else None

        # Task liveness gate (mirrors the interrupt handler's dual lookup):
        # the registration outlives agent_task by the SSE close window, and
        # run_conversation's finalizer + _push_steer_dropped_if_any have
        # already run by then — a steer stashed now would neither reach the
        # model nor produce a steer_dropped receipt (silently lost). Report
        # not_running so the caller re-queues the text as the next turn.
        task_done = False
        try:
            task_done = task is None or bool(task.done())
        except Exception:
            task_done = task is None
        if agent is None or task_done:
            return web.json_response(
                {"session_id": session_id, "status": "not_running", "accepted": False}
            )

        try:
            accepted = bool(agent.steer(text))
            # text is non-empty (validated above), so a normal False here
            # means the turn finalizer already closed the slot OR a hard
            # interrupt is winding the turn down (steer() refuses in the
            # stop window — an accepted steer there would be discarded by
            # the finalizer's interrupted branch with no receipt). Either
            # way the turn is effectively over: report not_running (the
            # contract's re-queue signal), not "rejected" — callers only
            # fall back to next-turn queueing on a re-queueable status.
            status = "steering" if accepted else "not_running"
        except Exception:
            logger.debug("[zet_agent] session steer: agent.steer failed", exc_info=True)
            accepted = False
            status = "rejected"

        return web.json_response(
            {"session_id": session_id, "status": status, "accepted": accepted}
        )

    def _interrupt_pending_interactions(self, session_id: str, scoped_session_key: Optional[str] = None) -> None:
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
        if scoped_session_key is None:
            scoped_session_key = self._active_turn_key(session_id)
        with self._clarify_state_lock:
            clarify_queue = list(self._clarify_queues.pop(scoped_session_key, []) or [])
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
            self._pending_clarify.pop(scoped_session_key, None)
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
            session_db = await self._ensure_session_db_async()
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
            if inspect.isawaitable(db_rows_cleared):
                # gw._session_db is the AsyncSessionDB facade (gateway/run.py
                # wraps SessionDB so SQLite never blocks the event loop): its
                # methods return coroutines. Without this await the clear was
                # a silent no-op AND the coroutine leaked into the JSON
                # response (TypeError → 500) — ZET-1139 regression class.
                db_rows_cleared = await db_rows_cleared
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
        reload_barrier_key, reload_barrier_owners = (
            self._snapshot_runtime_import_reload_barriers(
                _request_value(request, "hermes_profile_home")
            )
        )

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
            session_db = await self._ensure_session_db_async()
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
            if inspect.isawaitable(db_rows_cleared):
                # Same AsyncSessionDB facade as skills-reload above: await or
                # the clear silently no-ops and the coroutine breaks the JSON.
                db_rows_cleared = await db_rows_cleared
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
        self._release_runtime_import_reload_barriers(
            reload_barrier_key, reload_barrier_owners
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
        active_imports, unload_barrier_owner = self._block_runtime_import_profile(
            profile_home
        )
        if active_imports:
            return web.json_response(
                {
                    "unloaded": False,
                    "error": "profile has active imports",
                    "active_sessions": active_imports,
                    "active_imports": active_imports,
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
            except asyncio.CancelledError:
                self._unblock_runtime_import_profile(
                    profile_home, unload_barrier_owner
                )
                raise
            except Exception:
                self._unblock_runtime_import_profile(
                    profile_home, unload_barrier_owner
                )
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
            self._unblock_runtime_import_profile(
                profile_home, unload_barrier_owner
            )
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
                if db is self._session_db:
                    self._session_db = None

                async def _close_detached_session_db() -> None:
                    close = getattr(db, "close", None)
                    if not callable(close):
                        return
                    try:
                        await _to_thread_with_completion_barrier(close)
                    except Exception:
                        logger.warning(
                            "[zet_agent] profile-unload: SessionDB close failed",
                            exc_info=True,
                        )

                discard_staging = getattr(db, "discard_runtime_import_staging", None)
                if callable(discard_staging):
                    try:
                        await _to_thread_with_completion_barrier(discard_staging)
                    except asyncio.CancelledError as cancelled:
                        # The discard worker has finished before this branch is
                        # entered. Close in a second completion barrier, then
                        # release the profile barrier and preserve cancellation.
                        try:
                            await _close_detached_session_db()
                        except asyncio.CancelledError:
                            # A repeated cancellation is delivered only after
                            # the close worker has completed.
                            pass
                        self._unblock_runtime_import_profile(
                            profile_home, unload_barrier_owner
                        )
                        raise cancelled
                    except Exception:
                        # Unload remains best-effort, but always attempt to
                        # remove unpublished external transcripts before the
                        # DB leaves the background sweeper's cache.
                        logger.warning(
                            "[zet_agent] profile-unload: runtime import staging cleanup failed",
                            exc_info=True,
                        )
                try:
                    await _close_detached_session_db()
                except asyncio.CancelledError:
                    # _to_thread_with_completion_barrier has already observed
                    # close completion, so it is now safe to release unload.
                    self._unblock_runtime_import_profile(
                        profile_home, unload_barrier_owner
                    )
                    raise
                closed_session_db = True
            # 该 profile 的 goal barrier timers 一并取消（codex P1）：daemon
            # Timer 携带旧 profile 的 runtime scope，卸载后触发会用内存旧
            # scope 读 goal 并重新自驱一个用户刚删掉的 agent。同时按 home
            # 翻代，让还在跑的 post-turn judge 任务在 report 前的复核中失效
            # （active-run 计数在 turn 结束时已归零，拦不住这些后置任务）。
            # getattr：teardown 期间绝不懒创建 driver。
            drv = getattr(self, "_zet_goal_driver", None)
            if drv is not None:
                try:
                    drv.cancel_barrier_timers_for_home(profile_home)
                    drv.bump_lock_generations_for_home(profile_home)
                    # goal sidecar 走 hermes_cli.goals._DB_CACHE（按 home 缓存
                    # SessionDB），上面只关了 adapter 自己的 _session_dbs ——
                    # 不关它的话 profile 删除/重建后 goal 读写仍打在旧 inode
                    # 上（codex P1）。
                    drv.close_goal_db_for_home(profile_home)
                except Exception:
                    logger.warning(
                        "[zet_agent] profile-unload: goal timer cleanup failed",
                        exc_info=True,
                    )

        self._complete_runtime_import_profile_unload(
            profile_home, unload_barrier_owner
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
            self._app.router.add_get("/api/jobs/occurrences", self._handle_list_job_occurrences)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            self._app.router.add_post(
                "/api/sessions/import", self._handle_session_import,
            )
            self._app.router.add_post(
                "/api/memory/import", self._handle_memory_import,
            )
            self._register_unprefixed_cron_control_routes(self._app.router)
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
            self._app.router.add_post(
                "/v1/sessions/{session_id}/attachment/action",
                self._handle_attachment_action,
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
                "/v1/sessions/{session_id}/steer",
                self._handle_session_steer,
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
            # Delegation control plane (App banner status / cancel) — native
            # + profile-scoped mirrors, ZET_AGENT_KEY Bearer auth like the
            # rest of the surface.
            self._app.router.add_get(
                "/v1/delegations/status", self._handle_delegations_status
            )
            self._app.router.add_post(
                "/v1/delegations/{delegation_id}/cancel",
                self._handle_delegation_cancel,
            )
            self._app.router.add_post(
                "/v1/subagents/{subagent_id}/interrupt",
                self._handle_subagent_interrupt,
            )
            self._app.router.add_get(
                "/p/{profile}/v1/delegations/status",
                self._profile_handler(self._handle_delegations_status),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/delegations/{delegation_id}/cancel",
                self._profile_handler(self._handle_delegation_cancel),
            )
            self._app.router.add_post(
                "/p/{profile}/v1/subagents/{subagent_id}/interrupt",
                self._profile_handler(self._handle_subagent_interrupt),
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
                "/p/{profile}/v1/sessions/{session_id}/attachment/action",
                self._profile_handler(self._handle_attachment_action),
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
            self._app.router.add_post(
                "/p/{profile}/v1/sessions/{session_id}/steer",
                self._profile_handler(self._handle_session_steer),
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

            import_cleanup_task = asyncio.create_task(
                self._sweep_stale_runtime_imports()
            )
            self._background_tasks.add(import_cleanup_task)
            import_cleanup_task.add_done_callback(self._background_tasks.discard)

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
            # 已进 executor 的 post-turn judge 线程躲得过
            # cancel_background_tasks（取消的只是 asyncio wrapper）——
            # 全量翻代让它们在 report 前的复核中失效，否则会与替换者
            # （新 adapter reconcile / keepalive 拉起的新进程）并发
            # 自驱同一 goal（codex P1）。
            try:
                drv.invalidate_all_generations()
            except Exception:
                pass

        # Drop approval notify callbacks so blocked agent threads (if
        # any leak past process shutdown) don't fire into a dead loop.
        try:
            from tools.approval import unregister_gateway_notify
            with self._session_lock:
                sids = list(self._approval_session_ids)
                self._approval_session_ids.clear()
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
