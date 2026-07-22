"""Discover reusable Agent, Skill, and Task opportunities without interrupting work.

The plugin separates three product controls that should not be conflated:

* evaluation cadence: first turn, every third turn, plus optional main-model calls;
* candidate quality: a zero-shot semantic rubric with a structured ``none`` outcome;
* display cadence: cooldown, semantic deduplication, and dismissal latching.

The user's current task always remains the primary response.  A valid candidate is
rendered as a backwards-compatible recommendation envelope: new Zettlab clients
show a card, while older clients see the enclosed plain-text fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any

from gateway.response_filters import is_intentional_silence_response
from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

TOOL_NAME = "detect_creation_opportunity"
PLUGIN_VERSION = "0.6.0"
MIN_CONFIDENCE = 0.55
PROPOSAL_TTL_SECONDS = 30 * 60
DISMISS_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RECENT_PROPOSALS = 128
MAX_DISMISSALS = 128
EVALUATION_INTERVAL_TURNS = 3
PROMPT_COOLDOWN_TURNS = 10
SESSION_STATE_TTL_SECONDS = 24 * 60 * 60
MAX_SESSION_STATES = 512
CREATION_TYPES = {"agent", "skill", "task"}

_recent_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_dismissed_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_session_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
_state_lock = threading.Lock()
_plugin_llm: Any = None

_SELF_QUERY_RE = re.compile(
    r"(?:creation[\s_-]*governor|detect_creation_opportunity|propose_creation)",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_DISMISS_RE = re.compile(
    r"(?:暂时不要|不用创建|不要创建|先不创建|dismiss|not now|no thanks|do not create)",
    re.IGNORECASE,
)
_ACCEPT_RE = re.compile(
    r"(?:创建|开始创建|就这个|create it|create this|yes[, ]+create)",
    re.IGNORECASE,
)


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normalize_creation_type(value: Any) -> str:
    normalized = _text(value, 40).lower().replace("-", "_")
    if normalized == "scheduled_task":
        return "task"
    return normalized


def _semantic_dedup_key(value: Any, creation_type: str, suggested_name: str) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    slug = re.sub(r"[^a-z0-9._:-]+", "-", raw).strip("-")
    prefix = creation_type if creation_type in CREATION_TYPES else "proposal"
    if slug:
        if not slug.startswith(f"{prefix}:"):
            slug = f"{prefix}:{slug}"
        return slug[:120]

    fallback = unicodedata.normalize("NFKC", suggested_name).strip().casefold()
    digest = hashlib.sha256(f"{prefix}:{fallback}".encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _prune_timed_map(
    values: OrderedDict[tuple[str, str], float],
    *,
    expired_before: float,
    max_items: int,
) -> None:
    for key, created_at in tuple(values.items()):
        if created_at < expired_before:
            values.pop(key, None)
    while len(values) > max_items:
        values.popitem(last=False)


def _claim_proposal(session_id: str, dedup_key: str, now: float) -> bool:
    identity = (session_id, dedup_key)
    with _state_lock:
        _prune_timed_map(
            _recent_proposals,
            expired_before=now - PROPOSAL_TTL_SECONDS,
            max_items=MAX_RECENT_PROPOSALS,
        )
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return False
        _recent_proposals[identity] = now
        _prune_timed_map(
            _recent_proposals,
            expired_before=now - PROPOSAL_TTL_SECONDS,
            max_items=MAX_RECENT_PROPOSALS,
        )
    return True


def _is_dismissed(session_id: str, dedup_key: str, now: float) -> bool:
    identity = (session_id, dedup_key)
    with _state_lock:
        _prune_timed_map(
            _dismissed_proposals,
            expired_before=now - DISMISS_TTL_SECONDS,
            max_items=MAX_DISMISSALS,
        )
        return identity in _dismissed_proposals


def _latch_dismissal(session_id: str, dedup_key: str, now: float) -> None:
    with _state_lock:
        _dismissed_proposals[(session_id, dedup_key)] = now
        _prune_timed_map(
            _dismissed_proposals,
            expired_before=now - DISMISS_TTL_SECONDS,
            max_items=MAX_DISMISSALS,
        )


def _session_key(kwargs: dict[str, Any]) -> str:
    return _text(kwargs.get("session_id") or kwargs.get("task_id"), 160)


def _scoped_session_key(raw_session_id: str, owner_id: str) -> str:
    profile = str(get_hermes_home().resolve())
    return f"{profile}|{_text(owner_id, 160)}|{raw_session_id}"


def _session_key(kwargs: dict[str, Any]) -> str:
    raw_session_id = _raw_session_key(kwargs)
    if not raw_session_id:
        return ""
    invocation = _invocation_scope.get()
    profile_prefix = f"{get_hermes_home().resolve()}|"
    owner_fields = ("sender_id", "owner_id", "user_id")
    explicit_owner = next(
        (_text(kwargs.get(field), 160) for field in owner_fields if kwargs.get(field)),
        "",
    )
    if (
        invocation is not None
        and invocation[1].startswith(profile_prefix)
        and (not explicit_owner or explicit_owner == invocation[3])
    ):
        return invocation[1]
    if any(field in kwargs for field in owner_fields):
        return _scoped_session_key(raw_session_id, explicit_owner)
    return _scoped_session_key(raw_session_id, "")


def _prune_session_states(now: float) -> None:
    expired_before = now - SESSION_STATE_TTL_SECONDS
    for key, state in tuple(_session_states.items()):
        if float(state.get("last_seen", 0)) < expired_before:
            _session_states.pop(key, None)
    while len(_session_states) > MAX_SESSION_STATES:
        _session_states.popitem(last=False)


def _state_locked(session_id: str, now: float) -> dict[str, Any]:
    _prune_session_states(now)
    state = _session_states.get(session_id)
    if state is None:
        state = {
            "turn": 0,
            "last_evaluation_turn": 0,
            "last_prompt_turn": -10_000,
            "last_delivery_turn": -10_000,
            "last_candidate": None,
            "last_proposal": None,
            "proposal_stage": None,
            "draft_only_turn": None,
            "draft_delivered_turn": None,
            "awaiting_proposal_id": None,
            "authorized_turn": None,
            "native_bypass_turn": None,
            "last_user_message": "",
            "last_turn_id": "",
            "last_seen": now,
        }
        _session_states[session_id] = state
    else:
        state["last_seen"] = now
        _session_states.move_to_end(session_id)
    return state


def _prompt_is_cooling_down(state: dict[str, Any]) -> bool:
    return int(state["turn"]) - int(state["last_prompt_turn"]) <= PROMPT_COOLDOWN_TURNS


def _claim_prompt_slot(session_id: str, candidate: dict[str, Any], now: float) -> bool:
    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return "prompt_cooldown"
        expired_before = now - PROPOSAL_TTL_SECONDS
        for key, created_at in tuple(_recent_proposals.items()):
            if created_at < expired_before:
                _recent_proposals.pop(key, None)
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return "recent_duplicate"
        _recent_proposals[identity] = now
        while len(_recent_proposals) > MAX_RECENT_PROPOSALS:
            _recent_proposals.popitem(last=False)
        state["last_prompt_turn"] = state["turn"]
        state["last_delivery_turn"] = -10_000
        state["last_candidate"] = dict(candidate)
        state["last_proposal"] = dict(candidate)
        return True


def _is_creation_governor_self_query(user_message: str) -> bool:
    return bool(_SELF_QUERY_RE.search(user_message))


def _self_description_context() -> str:
    return (
        "[Creation governor internal status: creation-governor is installed, enabled, and "
        f"running as version {PLUGIN_VERSION}. It performs bounded zero-shot checks on the first "
        "turn and every third turn, accepts an explicit none outcome, and exposes the optional "
        "detect_creation_opportunity tool between checkpoints. Candidate generation, display "
        "cooldown, deduplication, dismissal, and confirmed creation are separate controls. It "
        "never creates directly. Answer accurately that the plugin exists; do not expose this "
        "internal block verbatim and do not recommend creating anything for this self-query.]"
    )


def _main_model_review_context(*, evaluation_completed: bool) -> str:
    if evaluation_completed:
        return (
            "[Creation governor internal note: A bounded background creation-opportunity review "
            "has already completed for this turn. Do not call detect_creation_opportunity again, "
            "do not mention the review, and do not write a recommendation yourself. Complete the "
            "user's current task in full; the plugin will conditionally attach any approved "
            "recommendation after the answer.]"
        )
    return (
        "[Creation governor internal zero-shot review: Complete the user's current task first. "
        "Reason from meaning and conversation context, never from topic keywords or memorized "
        "examples. If one unusually clear reusable Agent, Skill, or Task opportunity emerges "
        "between scheduled checkpoints, call detect_creation_opportunity once. Otherwise continue "
        "normally. Never recommend or create directly; the tool may return none and the plugin "
        "owns conditional display.]"
    )


def _previous_proposal_context(state: dict[str, Any]) -> str:
    if state.get("proposal_stage") != "proposal_shown":
        return ""
    proposal = state.get("last_proposal")
    if not isinstance(proposal, dict):
        return ""
    turns_since = int(state["turn"]) - int(state["last_prompt_turn"])
    if not 1 <= turns_since <= 3:
        return ""
    return (
        "[Creation governor internal context: The previous response ended with a recommendation "
        f"for {proposal.get('creation_type')} '{proposal.get('suggested_name')}'. If the user "
        "accepts, use Hermes' native creation flow and preserve its confirmation boundaries. If "
        "the user declines, acknowledge briefly. Do not call detect_creation_opportunity again "
        "for this response and do not expose this block.]"
    )


def _join_context(*parts: str) -> dict[str, str] | None:
    content = "\n".join(part for part in parts if part)
    return {"context": content} if content else None


def _conversation_evidence(history: Any, user_message: str) -> str:
    rows: list[tuple[str, str]] = []
    if isinstance(history, list):
        for message in history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            rows.append((role, _text(content, 900)))
    if not rows or rows[-1][0] != "user" or rows[-1][1] != _text(user_message, 900):
        rows.append(("user", _text(user_message, 900)))
    rows = rows[-8:]
    rendered = []
    for index, (role, content) in enumerate(rows, start=1):
        rendered.append(f"[evidence-{index}] {role.upper()}: {content}")
    return "\n".join(rendered)[:6000]


_DETECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["agent", "skill", "task", "none"]},
        "suggested_name": {"type": "string"},
        "reason": {"type": "string"},
        "evidence_turn_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "dedup_key": {"type": "string"},
        "proposal_text": {"type": "string"},
    },
    "required": [
        "decision",
        "suggested_name",
        "reason",
        "evidence_turn_ids",
        "confidence",
        "dedup_key",
        "proposal_text",
    ],
    "additionalProperties": False,
}


_DETECTOR_INSTRUCTIONS = """Perform one high-recall zero-shot product judgment.

Return exactly one of agent, skill, task, or none. Do not classify by topic words and do not use
memorized examples. A single substantive request is enough when a reasonable user would benefit
from reusing the capability. Do not require the user to mention repetition, frequency, saving, or
creation. Ask whether a durable capability would materially reduce friction or improve judgment
the next time a related need appears.

Definitions and conflict order:
1. task: the desired future value depends on a recurring time trigger, event trigger, background
   monitoring, or repeated refresh of new information. A word such as 'today' that merely scopes
   the current data is not by itself a future trigger.
2. agent: future work needs a long-lived responsible role, retained domain context, judgment,
   autonomous choice among tools, decisions about the next step, or repeated interpretation of a
   changing real-world business domain, account, operation, project, or body of evidence.
3. skill: future inputs vary but a stable input-to-output method can be reused without an
   independent identity or durable state.
4. none: small talk, a trivial transformation, a low-value closed-world fact lookup, an explicit
   request to create/configure/schedule something through Hermes' native flow, or no reasonable
   reuse value.

High-recall boundary: a substantive request to inspect, compare, diagnose, research, optimize, or
make a judgment about an ongoing external work domain should normally be agent rather than none,
even on the first request and even when the requested snapshot is scoped to today/current/latest.
Choose none only when reuse value is genuinely absent, not merely unstated.

Judge reuse value separately from current execution availability. Missing authorization,
connectors, data, or tools may block today's execution but is not a reason to ignore a clear
long-term need. Recommend only the first-layer object the user most needs, never multiple objects.
Match the user's language. For a positive decision, provide a concise name, concrete reason,
one-sentence optional proposal_text asking whether to create it, confidence, a stable semantic
dedup_key, and evidence_turn_ids chosen only from the supplied labels. For none, use empty strings,
an empty evidence list, and confidence 0. Never claim anything was created."""


def _run_forced_evaluation(
    *,
    user_message: str,
    conversation_history: Any,
) -> dict[str, Any] | None:
    llm = _plugin_llm
    if llm is None:
        return None
    evidence = _conversation_evidence(conversation_history, user_message)
    try:
        result = llm.complete_structured(
            instructions=_DETECTOR_INSTRUCTIONS,
            input=[{"type": "text", "text": evidence}],
            json_schema=_DETECTOR_SCHEMA,
            schema_name="creation_opportunity",
            temperature=0.0,
            max_tokens=500,
            timeout=25.0,
            purpose="creation_opportunity_checkpoint",
        )
        return result.parsed if isinstance(result.parsed, dict) else None
    except Exception as structured_error:
        error_text = str(structured_error).casefold()
        response_format_unavailable = "response_format" in error_text and any(
            marker in error_text
            for marker in ("unavailable", "unsupported", "not support", "invalid")
        )
        if not response_format_unavailable:
            logger.warning("creation opportunity checkpoint failed", exc_info=True)
            return None

    # Some OpenAI-compatible providers reject response_format even though the same
    # model can reliably return JSON from an ordinary bounded completion. Keep the
    # product check working without weakening local validation or exposing raw text.
    logger.info(
        "creation opportunity checkpoint provider lacks response_format; using plain JSON fallback"
    )
    try:
        result = llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        _DETECTOR_INSTRUCTIONS
                        + "\n\nReturn only one compact JSON object with exactly these keys: "
                        "decision, suggested_name, reason, evidence_turn_ids, confidence, "
                        "dedup_key, proposal_text. Do not use Markdown fences."
                    ),
                },
                {"role": "user", "content": evidence},
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=25.0,
            purpose="creation_opportunity_checkpoint_json_fallback",
        )
        parsed = _parse_detector_json(result.text)
        logger.info(
            "creation opportunity fallback decision=%s confidence=%s title=%s",
            parsed.get("decision") if parsed else None,
            parsed.get("confidence") if parsed else None,
            _text(parsed.get("suggested_name"), 80) if parsed else "",
        )
        return parsed
    except Exception:
        logger.warning("creation opportunity JSON fallback failed", exc_info=True)
        return None


def _parse_detector_json(value: Any) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_candidate(
    args: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    decision = _normalize_creation_type(
        args.get("decision") or args.get("creation_type")
    )
    if decision == "none":
        return None, "none"
    if decision not in CREATION_TYPES:
        return None, "unsupported_creation_type"

    suggested_name = _text(args.get("suggested_name"), 80)
    reason = _text(args.get("reason"), 400)
    proposal_text = _text(args.get("proposal_text"), 500)
    try:
        confidence = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < MIN_CONFIDENCE:
        return None, "confidence_below_threshold"
    if not suggested_name or not reason or not proposal_text:
        return None, "missing_candidate_fields"

    evidence_turn_ids = args.get("evidence_turn_ids")
    if not isinstance(evidence_turn_ids, list):
        evidence_turn_ids = []
    evidence_turn_ids = [
        _text(value, 80) for value in evidence_turn_ids[:8] if _text(value, 80)
    ]
    dedup_key = _semantic_dedup_key(args.get("dedup_key"), decision, suggested_name)
    return {
        "creation_type": decision,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence_turn_ids": evidence_turn_ids,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
        "current_request": _text(state.get("last_user_message"), 1000),
        "source_turn_id": _text(state.get("last_turn_id"), 160),
    }, "candidate"


def _proposal_payload(candidate: dict[str, Any], *, status: str) -> dict[str, Any]:
    current_request = (
        candidate.get("current_request") or "the current request already in context"
    )
    return {
        "status": status,
        "decision": candidate["creation_type"],
        "suggested_name": candidate["suggested_name"],
        "delivery": "deferred_to_transform_hook"
        if status == "proposal_ready"
        else "not_displayed",
        "next_step": (
            "Now complete the user's current task in full. A placeholder such as 'done' or "
            "'ready' is not a deliverable. Do not mention, quote, or paraphrase the creation "
            f"candidate. Current request: {current_request}"
        ),
    }


def _consider_candidate(
    session_id: str, args: dict[str, Any], now: float
) -> dict[str, Any]:
    with _state_lock:
        state = _state_locked(session_id, now)
        candidate, reason = _normalize_candidate(args, state)
        state["last_candidate"] = dict(candidate) if candidate else None
    if candidate is None:
        return {
            "status": "no_candidate" if reason == "none" else "not_proposed",
            "reason": reason,
        }

    if _is_dismissed(session_id, candidate["dedup_key"], now):
        return {"status": "candidate_recorded", "reason": "dismissed"}
    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return _proposal_payload(candidate, status="candidate_recorded") | {
                "reason": "prompt_cooldown"
            }
    if not _claim_proposal(session_id, candidate["dedup_key"], now):
        return {"status": "candidate_recorded", "reason": "recent_duplicate"}
    if not _claim_prompt_slot(session_id, candidate, now):
        return _proposal_payload(candidate, status="candidate_recorded") | {
            "reason": "prompt_cooldown"
        }
    return _proposal_payload(candidate, status="proposal_ready")


def _handle_previous_proposal_action(
    session_id: str, user_message: str, now: float
) -> str:
    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        if not isinstance(proposal, dict):
            return ""
        name = _text(proposal.get("suggested_name"), 80)
        dedup_key = _text(proposal.get("dedup_key"), 160)
    if name and name.casefold() not in user_message.casefold():
        return ""
    if _DISMISS_RE.search(user_message):
        if dedup_key:
            _latch_dismissal(session_id, dedup_key, now)
        with _state_lock:
            state = _state_locked(session_id, now)
            state["last_proposal"] = None
        return (
            "[Creation governor internal action: The user dismissed the previous recommendation. "
            "Acknowledge briefly, do not create anything, and do not run another opportunity "
            "review this turn.]"
        )
    if _ACCEPT_RE.search(user_message):
        return (
            "[Creation governor internal action: The user accepted the previous recommendation. "
            "Continue through Hermes' native creation flow, preserving its normal clarification "
            "and confirmation boundaries. Do not run another opportunity review this turn.]"
        )
    return ""


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    session_id = _session_key(kwargs)
    if not session_id:
        return None
    user_message = _text(kwargs.get("user_message"), 2000)
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        state["turn"] += 1
        state["last_user_message"] = user_message
        state["last_turn_id"] = _text(kwargs.get("turn_id"), 160)
        carry_context = _previous_proposal_context(state)
        turn = int(state["turn"])

    if _is_creation_governor_self_query(user_message):
        return _join_context(_self_description_context())

    action_context = _handle_previous_proposal_action(session_id, user_message, now)
    if action_context:
        return _join_context(action_context)

    evaluation_due = turn == 1 or turn % EVALUATION_INTERVAL_TURNS == 0
    if evaluation_due:
        with _state_lock:
            _state_locked(session_id, now)["last_evaluation_turn"] = turn
        candidate = _run_forced_evaluation(
            user_message=user_message,
            conversation_history=kwargs.get("conversation_history"),
        )
        if candidate is not None:
            _consider_candidate(session_id, candidate, now)
            return _join_context(
                carry_context, _main_model_review_context(evaluation_completed=True)
            )
        logger.info(
            "creation opportunity checkpoint unavailable; falling back to main-model review"
        )

    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return _join_context(carry_context)
    return _join_context(
        carry_context, _main_model_review_context(evaluation_completed=False)
    )


def _encode_recommendation(candidate: dict[str, Any]) -> str:
    payload = {
        "version": 1,
        "type": "creation_recommendation",
        "creation_type": candidate["creation_type"],
        "title": candidate["suggested_name"],
        "reason": candidate["reason"],
        "dedup_key": candidate["dedup_key"],
        "confidence": candidate["confidence"],
        "evidence_turn_ids": candidate.get("evidence_turn_ids") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fallback_text(candidate: dict[str, Any]) -> str:
    creation_type = candidate["creation_type"]
    name = candidate["suggested_name"]
    reason = candidate["reason"]
    proposal_text = candidate["proposal_text"]
    chinese = bool(_CJK_RE.search(name + reason + proposal_text))
    if chinese:
        label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
        return f"💡 可以沉淀为一个 {label}\n\n**「{name}」**\n\n{reason}\n\n{proposal_text}"
    label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
    return f"💡 This could become a reusable {label}\n\n**{name}**\n\n{reason}\n\n{proposal_text}"


def _recommendation_envelope(candidate: dict[str, Any]) -> str:
    encoded = _encode_recommendation(candidate)
    return (
        f"<!--creation-recommendation:start {encoded}-->\n\n"
        f"{_fallback_text(candidate)}\n\n"
        "<!--creation-recommendation:end-->"
    )


def _transform_llm_output(**kwargs: Any) -> str | None:
    session_id = _session_key(kwargs)
    response_text = str(kwargs.get("response_text") or "")
    if (
        not session_id
        or not response_text
        or _is_noninteractive(kwargs)
        or _is_unsupported_runtime(kwargs)
        or kwargs.get("structured_output")
        or is_intentional_silence_response(response_text)
    ):
        return None
    if kwargs.get("failed") or kwargs.get("interrupted") or kwargs.get("completed") is False:
        with _recent_lock:
            state = _state_locked(session_id, time.monotonic())
            state["pending_proposal"] = None
            if state.get("proposal_stage") == "draft_generating":
                _clear_draft_state(state, stage="proposal_shown")
        return None
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        current_turn = int(state["turn"])
        if (
            not isinstance(proposal, dict)
            or int(state["last_prompt_turn"]) != current_turn
            or int(state["last_delivery_turn"]) == current_turn
        ):
            return None
        state["last_delivery_turn"] = current_turn

    if "<!--creation-recommendation:start " in response_text:
        return None
    return response_text.rstrip() + "\n\n" + _recommendation_envelope(proposal)


def _detect_creation_opportunity(args: dict[str, Any], **kwargs: Any) -> str:
    session_id = _session_key(kwargs)
    if not session_id:
        return json.dumps({"status": "invalid", "error": "missing_session_id"})
    result = _consider_candidate(session_id, args, time.monotonic())
    return json.dumps(result, ensure_ascii=False)


def _on_pre_tool_call(**kwargs: Any) -> dict[str, str] | None:
    """Mechanically prevent creation side effects during the draft-only turn."""
    session_id = _session_key(kwargs)
    if not session_id:
        return None
    with _recent_lock:
        state = _state_locked(session_id, time.monotonic())
        if int(state.get("draft_only_turn") or -1) != int(state["turn"]):
            return None
    return {
        "action": "block",
        "message": "Creation is blocked while presenting the draft. Wait for explicit '确认创建'.",
    }


def _reset_state_for_tests() -> None:
    global _plugin_llm
    with _state_lock:
        _recent_proposals.clear()
        _dismissed_proposals.clear()
        _session_states.clear()
    _plugin_llm = None


def register(ctx: Any) -> None:
    global _plugin_llm
    try:
        _plugin_llm = ctx.llm
    except Exception:
        _plugin_llm = None

    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_tool(
        name=TOOL_NAME,
        toolset="creation_governor",
        schema={
            "name": TOOL_NAME,
            "description": (
                "Perform one high-recall zero-shot semantic choice among Agent, Skill, Task, and "
                "none. A single substantive request is enough; never require the user to mention "
                "repetition, saving, or creation. Use "
                "meaning and conversation context, never topic keyword matching or memorized "
                "examples. Task means desired future time/event/background execution; a current "
                "data range such as today is not by itself a trigger. Agent means a long-lived "
                "responsible role with retained context, judgment, autonomous tool choice, or "
                "interpretation of a changing real-world work domain. A substantive first request "
                "to inspect, compare, diagnose, research, optimize, or make a judgment about an "
                "ongoing external work domain should normally be Agent rather than none. "
                "Skill means a stable reusable input-to-output method without an independent "
                "identity. Missing connectors or authorization affect current execution, not "
                "reuse value. Explicit creation requests use Hermes' native flow and return none. "
                "Call only between scheduled checkpoints when one unusually clear opportunity "
                "emerges. The tool never creates and may return none. The plugin owns cooldown, "
                "deduplication, dismissal, and conditional card/text delivery after the current "
                "task is complete."
            ),
            "parameters": _DETECTOR_SCHEMA,
        },
        handler=_detect_creation_opportunity,
        description="Detect one reusable creation opportunity or none",
        emoji="💡",
    )
