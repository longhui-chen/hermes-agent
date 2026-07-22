"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature that AIAgent expects
for its callbacks. Internally, the callbacks push ACP session updates
to the client via ``conn.session_update()`` using
``asyncio.run_coroutine_threadsafe()`` (since AIAgent runs in a worker
thread while the event loop lives on the main thread).
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp
from acp.schema import AgentPlanUpdate, PlanEntry

from .tools import (
    build_tool_complete,
    build_tool_start,
    make_tool_call_id,
)

logger = logging.getLogger(__name__)

CONFIRMED_UPDATE_TIMEOUT_SECONDS = 5.0
MAX_PENDING_MESSAGE_UPDATES = 64


def _json_loads_maybe_prefix(value: str) -> Any:
    """Parse a JSON object even when Hermes appended a human hint after it."""
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)
        return data


def _build_plan_update_from_todo_result(result: Any) -> AgentPlanUpdate | None:
    """Translate Hermes' todo tool result into ACP's native plan update.

    Zed renders ``sessionUpdate: plan`` as its first-class task/todo panel. The
    Hermes agent already maintains task state through the ``todo`` tool, so the
    ACP adapter should expose that state natively instead of only as a generic
    tool-call transcript block.
    """
    if not isinstance(result, str) or not result.strip():
        return None

    try:
        data = _json_loads_maybe_prefix(result)
    except Exception:
        return None

    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None

    todos = data["todos"]
    if not todos:
        return AgentPlanUpdate(session_update="plan", entries=[])

    status_map = {
        "pending": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
        # ACP plans only support pending/in_progress/completed. Preserve
        # cancelled tasks as terminal entries instead of dropping them and
        # making the client's full-list replacement lose visible context.
        "cancelled": "completed",
    }
    entries: list[PlanEntry] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("id") or "").strip()
        if not content:
            continue
        raw_status = str(item.get("status") or "pending").strip()
        status = status_map.get(raw_status, "pending")
        if raw_status == "cancelled":
            content = f"[cancelled] {content}"
        entries.append(PlanEntry(content=content, priority="medium", status=status))

    return AgentPlanUpdate(session_update="plan", entries=entries)


async def _send_ordered_update(
    previous: Future | None,
    conn: acp.Client,
    session_id: str,
    update: Any,
) -> bool:
    """Send one append-only chunk after its predecessor succeeds."""
    if previous is not None:
        try:
            if await asyncio.wrap_future(previous) is False:
                return False
        except BaseException:
            return False
    try:
        await conn.session_update(session_id, update)
        return True
    except Exception:
        logger.debug("Failed to send ordered ACP message update", exc_info=True)
        return False


class ACPMessageDeliveryState:
    """Bounded sequential delivery state for one append-only ACP response."""

    def __init__(self, max_pending: int = MAX_PENDING_MESSAGE_UPDATES):
        self.max_pending = max_pending
        self._pending: Deque[tuple[Future, str]] = deque()
        self._prefix_chars = 0
        self._prefix_hash = hashlib.sha256()
        self._failed = False
        self._overflowed = False

    def _consume_completed(self) -> None:
        while self._pending and self._pending[0][0].done():
            future, text = self._pending.popleft()
            try:
                delivered = future.result() is not False
            except BaseException:
                delivered = False
            if self._failed or not delivered:
                self._failed = True
                continue
            self._prefix_chars += len(text)
            self._prefix_hash.update(text.encode("utf-8"))

    def enqueue(
        self,
        conn: acp.Client,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
        update: Any,
        text: str,
    ) -> bool:
        from agent.async_utils import safe_schedule_threadsafe

        self._consume_completed()
        if self._failed or self._overflowed:
            return False
        if len(self._pending) >= self.max_pending:
            self._overflowed = True
            return False
        previous = self._pending[-1][0] if self._pending else None
        future = safe_schedule_threadsafe(
            _send_ordered_update(previous, conn, session_id, update),
            loop,
            logger=logger,
            log_message="Failed to schedule ACP message update",
        )
        if future is None:
            self._failed = True
            return False
        self._pending.append((future, text))
        return True

    def wait_sync(self, timeout: float = CONFIRMED_UPDATE_TIMEOUT_SECONDS) -> bool:
        """Wait from the agent worker for the bounded ordered tail."""
        deadline = time.monotonic() + timeout
        while self._pending:
            future = self._pending[-1][0]
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeoutError:
                for pending, _text in self._pending:
                    pending.cancel()
                self._failed = True
                return False
            except BaseException:
                self._failed = True
                return False
            self._consume_completed()
        return not self._failed and not self._overflowed

    async def finish(self, timeout: float = CONFIRMED_UPDATE_TIMEOUT_SECONDS) -> None:
        """Settle or cancel the ordered tail before final fallback."""
        futures = [future for future, _text in self._pending]
        if futures:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(asyncio.wrap_future(future) for future in futures),
                        return_exceptions=True,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                for future in futures:
                    future.cancel()
                self._failed = True
                await asyncio.sleep(0)
        self._consume_completed()

    def remaining_content(self, final_response: str) -> str:
        """Return only the suffix not already confirmed by append-only ACP."""
        self._consume_completed()
        if self._prefix_chars <= 0:
            return final_response
        if self._prefix_chars > len(final_response):
            return final_response
        prefix = final_response[: self._prefix_chars]
        if hashlib.sha256(prefix.encode("utf-8")).digest() != self._prefix_hash.digest():
            return final_response
        return final_response[self._prefix_chars :]


def _send_update(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    update: Any,
) -> bool:
    """Schedule a fire-and-forget ACP update."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.session_update(session_id, update),
        loop,
        logger=logger,
        log_message="Failed to send ACP update",
    )
    if future is None:
        return False
    return future is not None


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
    edit_approval_policy_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and tracks IDs in a FIFO
    queue per tool name so duplicate/parallel same-name calls still complete
    against the correct ACP tool call.  Other event types (``tool.completed``,
    ``reasoning.available``) are silently ignored.
    """

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        # Only emit ACP ToolCallStart for tool.started; ignore other event types
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = make_tool_call_id()
        queue = tool_call_ids.get(name)
        if queue is None:
            queue = deque()
            tool_call_ids[name] = queue
        elif isinstance(queue, str):
            queue = deque([queue])
            tool_call_ids[name] = queue
        queue.append(tc_id)

        snapshot = None
        if name in {"write_file", "patch", "skill_manage"}:
            try:
                from agent.display import capture_local_edit_snapshot

                snapshot = capture_local_edit_snapshot(name, args)
            except Exception:
                logger.debug("Failed to capture ACP edit snapshot for %s", name, exc_info=True)
        tool_call_meta[tc_id] = {"args": args, "snapshot": snapshot}

        edit_diff = None
        if name in {"write_file", "patch"} and edit_approval_policy_getter is not None:
            try:
                from acp_adapter.edit_approval import build_edit_proposal, should_auto_approve_edit

                proposal = build_edit_proposal(name, args)
                if proposal is not None:
                    policy, cwd = edit_approval_policy_getter()
                    if should_auto_approve_edit(proposal, policy, cwd):
                        edit_diff = proposal
            except Exception:
                logger.debug("Failed to prepare auto-approved ACP edit diff for %s", name, exc_info=True)

        update = build_tool_start(tc_id, name, args, edit_diff=edit_diff)
        _send_update(conn, session_id, loop, update)

    return _tool_progress


# ------------------------------------------------------------------
# Thinking callback
# ------------------------------------------------------------------

def make_thinking_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""

    def _thinking(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_thought_text(text)
        _send_update(conn, session_id, loop, update)

    return _thinking


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
    tool_call_meta: Dict[str, Dict[str, Any]],
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if prev_tools and isinstance(prev_tools, list):
            for tool_info in prev_tools:
                tool_name = None
                result = None
                function_args = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                    function_args = tool_info.get("arguments") or tool_info.get("args")
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()
                    meta = tool_call_meta.pop(tc_id, {})
                    update = build_tool_complete(
                        tc_id,
                        tool_name,
                        result=str(result) if result is not None else None,
                        function_args=function_args or meta.get("args"),
                        snapshot=meta.get("snapshot"),
                    )
                    _send_update(conn, session_id, loop, update)
                    if tool_name == "todo":
                        plan_update = _build_plan_update_from_todo_result(result)
                        if plan_update is not None:
                            _send_update(conn, session_id, loop, plan_update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)

    return _step


# ------------------------------------------------------------------
# Agent message callback
# ------------------------------------------------------------------

def make_message_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    *,
    confirm_delivery: bool = False,
    delivery_state: ACPMessageDeliveryState | None = None,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""

    def _message(text: str) -> bool:
        if not text:
            return False
        update = acp.update_agent_message_text(text)
        if delivery_state is None:
            return _send_update(conn, session_id, loop, update)
        if not delivery_state.enqueue(conn, session_id, loop, update, text):
            return False
        if confirm_delivery:
            return delivery_state.wait_sync()
        return True

    return _message
