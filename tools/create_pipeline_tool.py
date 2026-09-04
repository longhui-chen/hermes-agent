"""Drive the platform's application-creation pipeline from the chat agent.

app-dispatch routes a NEW-app request here instead of the main Chat Agent
free-lancing skills (product-prototyping, app-coding, …). The platform owns the
whole flow as a state machine; this tool only exposes the guide-stage actions
the model is allowed to take, and every reply carries a `next` hint the model
should follow:

  start        open (or return) the owner's creation run for this chat session
               (idempotent: an unfinished run is returned with existing=true)
  set_pace     record the user's pace choice: "direct" | "ask"
  submit_spec  hand in the App Spec (JSON object); the platform validates it and
               either freezes it (awaiting the USER's confirmation) or returns
               structured `problems` to fix
  status       poll a run (step / guide_state / done / failed / entry_url)
  cancel       abandon before the build starts

There is deliberately NO confirm action: only the user starts the build (chat
text the platform recognizes, or the creation page button). Loopback-only, same
trust model as app_host. `create` is kept as an alias of `start` for older skill
text.

Registered under the zettlab_apphost toolset (verified on-device: a separate
toolset's tool was silently dropped by the platform reverse-mapping; app_host's
toolset resolves + carries the same App Host secret scope).
"""

from __future__ import annotations

import ipaddress
import json
from typing import Any, Dict
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from tools.registry import registry

_ACTIONS = {"start", "create", "set_pace", "submit_spec", "revise", "status", "cancel"}
_SESSION_HEADER = "X-Zettlab-Session-Id"
_SCHEMA = {
    "name": "create_pipeline",
    "description": (
        "Platform application-creation pipeline (the ONLY way to create a new app on "
        "this device; never write app code or open prototyping skills yourself). "
        "action='start' opens the creation run for this chat (needs `intent`, the user's "
        "one-line request; returns run_id + guide_state + next). Then follow `next` each "
        "turn: action='set_pace' (run_id, pace='direct'|'ask') after the user picks a pace; "
        "action='submit_spec' (run_id, spec=<App Spec JSON object>) when you have the "
        "requirements — the platform validates it and returns either an accepted spec "
        "(then ask the USER to reply 确认) or `problems` to fix; action='revise' (run_id, "
        "note=the user's change request in one sentence) when the user asks to change the "
        "requirements or the prototype before confirming — the platform merges it into the "
        "spec and updates the existing prototype in place; action='status' (run_id) to "
        "report progress; action='cancel' (run_id) if the user gives up. You cannot start "
        "the build yourself: only the user's confirmation does."
    ),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "intent": {"type": "string", "maxLength": 4096},
        "run_id": {"type": "string", "maxLength": 80},
        "pace": {"type": "string", "enum": ["direct", "ask"]},
        "spec": {"type": "object", "description": "App Spec JSON object (exact keys per the platform schema block)"},
        "note": {"type": "string", "maxLength": 2048, "description": "revise: the user's change request (their own words, or your one-sentence summary)"},
        "reason": {"type": "string", "maxLength": 512},
    }, "required": ["action"], "additionalProperties": False},
}


def _secret(name: str) -> str:
    return str(get_secret(name, "") or "").strip()


def _base_url() -> str:
    """Derive the create-pipeline loopback base from the app_host base
    (…/api/v1/internal/apphost -> …/api/v1/internal/createpipeline). Reuses the
    already-injected ZET_APPHOST_BASE_URL so no new registry env is needed."""
    raw = _secret("ZET_APPHOST_BASE_URL")
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or not parsed.port or not ipaddress.ip_address(host).is_loopback:
            return ""
    except (ValueError, TypeError):
        return ""
    path = parsed.path.rstrip("/")
    if path.endswith("/apphost"):
        path = path[: -len("/apphost")] + "/createpipeline"
    elif "/internal" in path:
        path = path.rsplit("/internal", 1)[0] + "/internal/createpipeline"
    else:
        path = path + "/createpipeline"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _cp_enabled() -> bool:
    # Gate on the App Host base secret + the action token (same session condition
    # as app_host). The /createpipeline sibling is always reachable when /apphost
    # is; do NOT gate on the derived URL (call-time concern). A unique name avoids
    # any registry check_fn keying collision with other tools' `_enabled`.
    return bool(_secret("ZET_APPHOST_BASE_URL") and _secret("ZETTLAB_AGENT_ACTION_TOKEN"))


_cp_enabled._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _session_id() -> str:
    """The chat session this turn runs in: local-server sends it to hermes as the
    stable session key (X-Hermes-Session-Key) and hermes freezes it for the turn
    (gateway.session_context.execution_session_key). The platform binds the
    creation run to it so it can cage exactly this session's turns."""
    try:
        from gateway.session_context import execution_session_key

        return str(execution_session_key() or "").strip()
    except Exception:  # noqa: BLE001 — never let a missing helper break the tool
        return ""


def _spec_object(raw: Any) -> Dict[str, Any] | None:
    """Accept the spec as a dict, or as a string holding a JSON object."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _result(body: Any, status_code: int) -> str:
    """Return the platform body verbatim (it already carries ok/code/next);
    non-JSON or empty bodies degrade to a structured error."""
    if isinstance(body, dict):
        if status_code >= 400 and "ok" not in body:
            body = {"ok": False, "code": "http_%d" % status_code, **body}
        return json.dumps(body, ensure_ascii=False)
    return json.dumps({"ok": False, "code": "http_%d" % status_code, "error": str(body)[:256]}, ensure_ascii=False)


def create_pipeline(args: Any = None, **_: Any) -> str:
    # hermes dispatch calls handler(args, **kwargs): the first positional `args`
    # is the parsed argument dict (see app_host_tool's args.get(...) pattern —
    # the unpacked-kwargs signature crashes with "unhashable type: 'dict'").
    if not isinstance(args, dict):
        args = {}
    action = str(args.get("action", "") or "").strip()
    if action not in _ACTIONS:
        return json.dumps({"ok": False, "success": False, "code": "invalid_action",
                           "next": "action 只能是 start / set_pace / submit_spec / status / cancel。"}, ensure_ascii=False)
    if action == "create":
        action = "start"
    base = _base_url()
    if not base:
        return json.dumps({"ok": False, "success": False, "code": "create_pipeline_unavailable",
                           "next": "设备的应用创建服务不可用，如实告诉用户稍后再试；不要自己写代码替代。"}, ensure_ascii=False)
    headers = {"X-Zettlab-Agent-Action-Token": _secret("ZETTLAB_AGENT_ACTION_TOKEN")}
    session_id = _session_id()
    if session_id:
        headers[_SESSION_HEADER] = session_id
    run_id = str(args.get("run_id", "") or "").strip()
    try:
        if action == "start" and not str(args.get("intent", "") or "").strip() and run_id:
            # 模型在用户确认那一轮常常习惯性再调一次 create/start(run_id)（09-03 药箱 #2 就是
            # 这样拿到裸错误后编出"已开始建造"）。带 run_id 不带 intent = 它其实想知道现状：
            # 直接当 status 查，把真实状态和 next 给回去，而不是一句没有指引的 intent_required。
            response = requests.get(base + "/status/" + quote(run_id, safe=""), headers=headers, timeout=10)
        elif action == "start":
            intent = str(args.get("intent", "") or "").strip()
            if not intent:
                return json.dumps({"ok": False, "success": False, "code": "intent_required",
                                   "next": "start 只在还没有 run 时用，必须带 intent（用户的一句话诉求）；已有 run 请用 status / set_pace / submit_spec / cancel 并带 run_id。你没有确认动作：用户的「确认」由平台处理。"},
                                  ensure_ascii=False)
            body: Dict[str, Any] = {"intent": intent}
            owner = _secret("ZET_AGENT_ID")
            if owner:
                body["owner_agent"] = owner
            if session_id:
                body["session_id"] = session_id
            response = requests.post(base + "/start", headers=headers, json=body, timeout=15)
        elif action == "status":
            if not run_id:
                return json.dumps({"ok": False, "success": False, "code": "run_id_required",
                                   "next": "status 需要 run_id（start 返回的 run_id）。"}, ensure_ascii=False)
            response = requests.get(base + "/status/" + quote(run_id, safe=""), headers=headers, timeout=10)
        else:
            if not run_id:
                return json.dumps({"ok": False, "success": False, "code": "run_id_required",
                                   "next": action + " 需要 run_id（start 返回的 run_id）。"}, ensure_ascii=False)
            if action == "set_pace":
                pace = str(args.get("pace", "") or "").strip().lower()
                if pace not in {"direct", "ask"}:
                    return json.dumps({"ok": False, "code": "invalid_pace", "next": "pace 只能是 direct 或 ask。"}, ensure_ascii=False)
                payload: Dict[str, Any] = {"run_id": run_id, "pace": pace}
                path = "/set-pace"
            elif action == "revise":
                note = str(args.get("note", "") or "").strip()
                if not note:
                    return json.dumps({"ok": False, "code": "note_required",
                                       "next": "revise 需要 note：用一句话说明用户要改什么（可直接用用户原话）。"}, ensure_ascii=False)
                payload = {"run_id": run_id, "note": note[:2048]}
                path = "/revise"
            elif action == "submit_spec":
                spec = _spec_object(args.get("spec"))
                if spec is None:
                    return json.dumps({"ok": False, "code": "spec_required",
                                       "next": "spec 必须是一个 JSON 对象（按平台给出的 App Spec 键名）。"}, ensure_ascii=False)
                payload = {"run_id": run_id, "spec": spec}
                path = "/submit-spec"
            else:  # cancel
                payload = {"run_id": run_id, "reason": str(args.get("reason", "") or "")[:512]}
                path = "/cancel"
            response = requests.post(base + path, headers=headers, json=payload, timeout=15)
        try:
            parsed = response.json()
        except ValueError:
            parsed = response.text
        return _result(parsed, response.status_code)
    except requests.RequestException as exc:
        return json.dumps({"ok": False, "success": False, "code": "create_pipeline_unavailable", "error": str(exc)[:256]}, ensure_ascii=False)


registry.register(name="create_pipeline", toolset="zettlab_apphost", schema=_SCHEMA, handler=create_pipeline, check_fn=_cp_enabled, defer_to_tool_search=False)
