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

import json
import logging
import os
import threading
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
    _coerce_port,
    _openai_error,
)
# ZettClaw cron event hook — monkey-patches cron.scheduler at import time
# so cron triggers POST a webhook to local-server. zero hermes main-line
# changes; see zet_agent_cron.py docstring for the full rationale.
# No-op when CRON_WEBHOOK_URL env var is unset (i.e. non-ZettClaw deploys).
from gateway.platforms import zet_agent_cron as _zet_agent_cron
_zet_agent_cron.install()

logger = logging.getLogger(__name__)

# Default port for the Zet Agent platform. Distinct from API_SERVER's
# 8642 so both platforms can run side-by-side during the migration.
ZET_AGENT_DEFAULT_PORT = 7900

# How long the agent thread will block waiting for a user response to
# a clarify prompt before giving up and returning an empty string. The
# approval flow uses hermes' built-in timeout (``approval.gateway_timeout``,
# default 300s) so we don't duplicate it here.
CLARIFY_RESPONSE_TIMEOUT = 300.0

# Cap the auto-title at a length the APP can render in a single line
# without truncation. Beyond that, the APP can elide.
TITLE_MAX_LEN = 60


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
        pattern_keys}`` — no request_id (per-session FIFO; oldest
        pending wins). expires_at_ms is injected by the local-server
        adapter so all clients see one timer source.

        We also cache the payload in ``self._pending_approval`` so a
        reconnecting client can fetch it via the GET /pending endpoint
        and re-render the modal after a ws drop.
        """
        def _notify(approval_data: Dict[str, Any]) -> None:
            payload = {
                "type": "hermes.approval",
                "command": approval_data.get("command", ""),
                "description": approval_data.get("description", ""),
                "pattern_key": approval_data.get("pattern_key", ""),
                "pattern_keys": list(approval_data.get("pattern_keys", []) or []),
            }
            with self._pending_lock:
                self._pending_approval[session_id] = payload
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] approval notify push failed", exc_info=True)

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

            payload = {
                "type": "hermes.clarify",
                "question": question,
                "choices_offered": list(choices or []),
            }
            with self._pending_lock:
                self._pending_clarify[session_id] = payload
            try:
                stream_q.put(("__tool_progress__", payload))
            except Exception:
                logger.debug("[zet_agent] clarify push failed", exc_info=True)
                self._discard_clarify_entry(session_id, entry)
                return ""

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
                    platform="zettlab",
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
        )

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

        # 2. Clarify: late-bind. The AIAgent invokes this only if the
        # model calls the clarify tool, so the cost of always wiring
        # it is just a closure allocation.
        if session_id:
            try:
                agent.clarify_callback = self._make_clarify_cb(stream_q, session_id)
            except Exception:
                logger.warning("[zet_agent] failed to attach clarify_callback", exc_info=True)

        # 3. Approval: register a per-session notify callback.
        # We don't unregister here because chat.completions reuses the
        # same session_id across turns; unregistration happens on
        # platform disconnect (or never, for short-lived processes).
        if session_id:
            try:
                from tools.approval import register_gateway_notify
                register_gateway_notify(session_id, self._make_approval_cb(stream_q, session_id))
            except Exception:
                logger.warning("[zet_agent] failed to register approval notify", exc_info=True)

        # 4. Auto-title is emitted in _run_agent() instead — the
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
        if stream_q is not None:
            try:
                title = self._maybe_update_title(session_id, user_message)
                self._push_title_if_new(stream_q, session_id, title)
            except Exception:
                logger.debug("[zet_agent] auto-title hook failed", exc_info=True)

        # Consume pending model-switch note (set by /v1/model/switch).
        # The main GatewayServer._dispatch_message checks this dict for
        # Telegram/Discord/etc, but the zet_agent /v1/chat/completions
        # path bypasses that dispatcher — so we consume it here instead.
        gw = getattr(self, "gateway_runner", None)
        if gw is not None and session_id:
            notes = getattr(gw, "_pending_model_notes", None)
            if notes:
                note = notes.pop(session_id, None)
                if not note and gateway_session_key:
                    note = notes.pop(gateway_session_key, None)
                if note and user_message:
                    user_message = note + "\n\n" + user_message

        old_session_key = os.environ.get("HERMES_SESSION_KEY")
        old_exec_ask = os.environ.get("HERMES_EXEC_ASK")
        if session_id:
            os.environ["HERMES_SESSION_KEY"] = session_id
        os.environ.setdefault("HERMES_EXEC_ASK", "1")

        try:
            return await super()._run_agent(
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
            )
        finally:
            if old_session_key is None:
                os.environ.pop("HERMES_SESSION_KEY", None)
            else:
                os.environ["HERMES_SESSION_KEY"] = old_session_key
            if old_exec_ask is None:
                os.environ.pop("HERMES_EXEC_ASK", None)
            else:
                os.environ["HERMES_EXEC_ASK"] = old_exec_ask

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
        """POST /v1/model/switch — agent-level model switch without restart.

        Called by zettlab-local-server's PUT /api/v1/agent/agents/:id/model
        endpoint. Updates the profile's config.yaml model.* slot AND
        live-swaps all cached AIAgent instances so existing sessions pick
        up the new model immediately (same mechanism as hermes' /model
        command, but applied to every session at once).

        Expected body: {"model": "...", "provider": "...", "base_url": "...", "api_key": "..."}
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)

        new_model = body.get("model", "")
        new_provider = body.get("provider", "")
        new_base_url = body.get("base_url", "")
        new_api_key = body.get("api_key", "")
        if not new_model:
            return web.json_response({"ok": False, "error": "model is required"}, status=400)

        # 1. Update profile config.yaml so the change persists across
        #    gateway restarts and new sessions read the right default.
        try:
            from gateway.run import _load_gateway_config, _hermes_home
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
            cfg["model"] = model_slot
            config_path = _hermes_home / "config.yaml"
            atomic_yaml_write(config_path, cfg)
        except Exception as exc:
            logger.warning("model-switch: config write failed: %s", exc)
            return web.json_response({"ok": False, "error": f"config write: {exc}"}, status=500)

        # 2. Live-swap all cached agents (mirrors hermes /model command).
        switched = 0
        gw = getattr(self, "gateway_runner", None)
        if gw is not None:
            cache_lock = getattr(gw, "_agent_cache_lock", None)
            cache = getattr(gw, "_agent_cache", None)
            if cache_lock is not None and cache is not None:
                with cache_lock:
                    entries = list(cache.items())
                for key, entry in entries:
                    agent = entry[0] if isinstance(entry, tuple) else entry
                    if agent is not None and hasattr(agent, "switch_model"):
                        try:
                            agent.switch_model(
                                new_model=new_model,
                                new_provider=new_provider,
                                api_key=new_api_key,
                                base_url=new_base_url,
                            )
                            switched += 1
                        except Exception as exc:
                            logger.warning("model-switch: agent swap failed for %s: %s", key, exc)

            # Clear session-level /model overrides — the agent-level switch
            # takes precedence; stale per-session overrides would shadow it.
            overrides = getattr(gw, "_session_model_overrides", None)
            if overrides is not None:
                overrides.clear()

            # Inject a model-switch note for every cached session so the
            # LLM knows the model changed on the next turn — same mechanism
            # hermes /model uses (gateway/run.py _pending_model_notes).
            # Without this, the LLM still has "I'm <old_model>" in its
            # conversation history and keeps self-identifying as the old
            # model until the context naturally rolls over.
            old_model = body.get("old_model", "")
            if not hasattr(gw, "_pending_model_notes"):
                gw._pending_model_notes = {}
            note = (
                f"[Note: model was just switched"
                f"{f' from {old_model}' if old_model else ''}"
                f" to {new_model}"
                f"{f' via {new_provider}' if new_provider else ''}."
                f" Adjust your self-identification accordingly.]"
            )
            for key in list(cache.keys()) if cache is not None else []:
                gw._pending_model_notes[key] = note

        logger.info("model-switch: model=%s provider=%s switched=%d agents", new_model, new_provider, switched)
        return web.json_response({"ok": True, "model": new_model, "switched": switched})

    # ------------------------------------------------------------------
    # connect — extend base routes with our respond endpoints
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp server, registering our extra routes
        before the base class sets up the runner (which freezes the
        router).

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
                "/v1/model/switch",
                self._handle_model_switch,
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
            return True
        except Exception:
            logger.exception("[%s] failed to start", self.name)
            return False

    async def disconnect(self) -> None:
        """Tear down the aiohttp server and unregister approval callbacks
        for any sessions we registered."""
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
