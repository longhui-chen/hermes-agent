#!/usr/bin/env python3
"""call_agent — hand a task to ANOTHER real agent on this device (zettlab).

Unlike ``delegate_task`` (anonymous throwaway workers cloned from the caller),
``call_agent`` reaches a REAL agent profile: the callee runs with its own
persona (SOUL.md), skills, memory, and model config, inside a dedicated A2A
session (``zettlab:<user>:<callee>:a2a-<caller>``) owned by local-server —
so the collaboration also becomes part of the callee's memory.

The tool itself is intentionally thin: one synchronous loopback POST to
local-server's agent-call endpoint. ALL guardrails live server-side where
the authority is (HR#3 fail-closed):

  - ownership: caller and callee must belong to the same user;
  - cycle/depth: local-server reconstructs the call chain from its A2A
    session registry (A→B→A rejected, chain depth capped);
  - concurrency quota + wall-clock timeout;
  - progress surfacing: local-server synthesizes delegation.status
    (kind=agent_call) events into the CALLER's in-flight turn, so the App
    banner and drill-down work with zero extra wiring here.

Availability is env-gated like every zettlab loopback integration
(ZET_AGENT_CALL_URL, written by local-server): without it the tool is not
even registered into the model's schema, so upstream/CLI deployments never
see it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_CALL_URL_ENV = "ZET_AGENT_CALL_URL"

# Wall-clock cap for one agent call. Mirrors the server-side cap; the client
# cap is slightly LONGER so the server's richer timeout error (with the
# callee's partial state) wins the race when both fire.
_CALL_TIMEOUT_SECONDS = 15 * 60 + 30


def _scoped_env(name: str, default: str = "") -> str:
    """Profile-scoped env resolution (mux-safe).

    Mirrors zet_agent_cron._scoped_env's fail-closed contract: under an
    ACTIVE multiplexer, a profile missing this variable must NOT fall back
    to os.environ — the process env belongs to the boot profile, so the
    fallback would let one profile issue call_agent with ANOTHER profile's
    identity (ZET_AGENT_ID / call URL). Only legacy single-profile
    processes may read the global environment.
    """
    try:
        from gateway.platforms.zet_agent_cron import _scoped_env as scoped

        value = scoped(name, "")
        if value:
            return value
    except Exception:
        pass
    try:
        from agent.secret_scope import is_multiplex_active

        if is_multiplex_active():
            return default
    except Exception:
        pass
    return os.environ.get(name, default)


def _call_url() -> str:
    return _scoped_env(_CALL_URL_ENV, "").strip()


def _own_session_id() -> str:
    """The caller's hermes session id, parsed from the gateway session_key.

    Two real-world shapes (mirrors local-server's
    hermesSessionIDFromSessionKey):

    - ``agent:main:zet_agent:<chat_type>:<chat_id>`` (cron-originated) —
      strip the known prefix, keep the rest (chat_id itself has colons);
    - ``zettlab:<user>:<agent>:<chat>`` — zet_agent chat turns bind the
      local-server session id verbatim; use it as-is.

    Empty for CLI/unknown contexts (the endpoint then rejects: no owner).
    """
    try:
        from tools.approval import get_current_session_key

        key = str(get_current_session_key("") or "")
    except Exception:
        return ""
    marker = ":zet_agent:"
    idx = key.find(marker)
    if idx < 0:
        if key.startswith("zettlab:") and key.count(":") >= 3:
            return key
        return ""
    rest = key[idx + len(marker):]
    sep = rest.find(":")
    if sep < 0:
        return ""
    return rest[sep + 1:]


def check_call_agent_requirements() -> bool:
    """Schema-gate: only devices where local-server published the endpoint."""
    return bool(_call_url())


def call_agent(agent: str = "", message: str = "", parent_agent=None) -> str:
    """Synchronously hand ``message`` to agent ``agent`` and return its reply."""
    agent = str(agent or "").strip()
    message = str(message or "").strip()
    if not agent or not message:
        return json.dumps(
            {"error": "call_agent requires both 'agent' and 'message'"},
            ensure_ascii=False,
        )
    url = _call_url()
    if not url:
        return json.dumps(
            {"error": "agent-to-agent calls are not available on this device"},
            ensure_ascii=False,
        )

    payload = {
        "schema": 1,
        "caller_agent_id": _scoped_env("ZET_AGENT_ID", "").strip(),
        "caller_session_id": _own_session_id(),
        "callee": agent,
        "message": message,
    }

    started = time.time()
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_CALL_TIMEOUT_SECONDS) as resp:
                body = resp.read()
        except urllib.error.HTTPError as http_exc:
            # 4xx/5xx 的响应体就是 local-server 的信封（cycle/depth/quota/
            # unauthorized/caller_cancelled + available_agents）。裸抛
            # "HTTP Error 403" 模型无法自纠——真机上它会连猜三个不存在的
            # agent 名字然后放弃。读出结构化原因走统一的字段透传。
            body = http_exc.read()
        result = json.loads(body.decode("utf-8"))
    except Exception as exc:
        logger.warning("call_agent(%s) transport failure: %s", agent, exc)
        return json.dumps(
            {
                "error": f"agent call failed: {type(exc).__name__}: {exc}",
                "agent": agent,
                "duration_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        )

    # local-server's envelope: {code, data:{...}}. Surface data verbatim-ish;
    # the model needs reply text on success and a machine-readable reason on
    # refusal (cycle / depth / quota / unauthorized / busy / timeout).
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, dict):
        data = result if isinstance(result, dict) else {}
    out: Dict[str, Any] = {
        "agent": agent,
        "duration_seconds": round(time.time() - started, 2),
    }
    for field in (
        "reply",
        "error",
        "reason",
        "available_agents",
        "agent_id",
        "agent_name",
        "session_id",
        "turn_id",
    ):
        value = data.get(field)
        if value is not None:
            out[field] = value
    if "reply" not in out and "error" not in out:
        out["error"] = "agent call returned no reply"
    return json.dumps(out, ensure_ascii=False)


CALL_AGENT_SCHEMA = {
    "name": "call_agent",
    "description": (
        "Hand a task or question to ANOTHER agent on this device and get its "
        "reply. The callee is a real agent with its own persona, skills, and "
        "memory — use this when the task needs that agent's expertise or "
        "context, and the collaboration should become part of its memory.\n\n"
        "Contrast with delegate_task: delegate_task spawns anonymous workers "
        "of YOURSELF for parallel throughput; call_agent consults a specific "
        "COLLEAGUE. Prefer delegate_task for parallelizable grunt work, "
        "call_agent when the other agent's identity matters.\n\n"
        "The call is synchronous and can take minutes for complex tasks. "
        "Include ALL context the other agent needs in 'message' — it cannot "
        "see this conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": (
                    "Target agent's name (as shown in the device's agent "
                    "list) or agent id."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The self-contained task or question for the target "
                    "agent, including all necessary context."
                ),
            },
        },
        "required": ["agent", "message"],
    },
}


registry.register(
    name="call_agent",
    toolset="agent_call",
    schema=CALL_AGENT_SCHEMA,
    handler=lambda args, **kw: call_agent(
        agent=args.get("agent", ""),
        message=args.get("message", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_call_agent_requirements,
    emoji="🤝",
)
