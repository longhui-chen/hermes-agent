"""Discover reusable capabilities through the main model's semantic judgment.

The plugin is a governor, not a classifier and not a creator.  It gives the
main Hermes model a zero-shot decision rubric, while deterministic code owns
cooldowns, duplicate suppression, validation, and confirmation boundaries.
Explicit creation requests remain on Hermes' native creation paths.
"""

from __future__ import annotations

import hashlib
import json
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


TOOL_NAME = "propose_creation"
PLUGIN_VERSION = "0.5.4"
MIN_CONFIDENCE = 0.55
PROPOSAL_TTL_SECONDS = 30 * 60
MAX_RECENT_PROPOSALS = 128
EVALUATION_INTERVAL_TURNS = 3
PROMPT_COOLDOWN_TURNS = 10
SESSION_STATE_TTL_SECONDS = 24 * 60 * 60
MAX_SESSION_STATES = 512
CREATION_TYPES = {"agent", "skill", "scheduled_task"}
UNSUPPORTED_API_MODES = {"codex_app_server"}
UNSUPPORTED_PLATFORMS = {"acp"}

_recent_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_session_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
_recent_lock = threading.Lock()
_invocation_scope: ContextVar[tuple[str, str, str | None, str] | None] = ContextVar(
    "creation_governor_invocation_scope",
    default=None,
)

_SELF_QUERY_RE = re.compile(
    r"(?:creation[\s_-]*governor|propose_creation)",
    re.IGNORECASE,
)


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _semantic_dedup_key(
    value: Any,
    creation_type: str,
    suggested_name: str,
) -> str:
    """Return a stable, non-empty key even when every display character is Unicode."""
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


def _claim_proposal(session_id: str, dedup_key: str, now: float) -> bool:
    """Return False when the same opportunity was proposed recently."""
    identity = (session_id, dedup_key)
    with _recent_lock:
        expired_before = now - PROPOSAL_TTL_SECONDS
        for key, created_at in tuple(_recent_proposals.items()):
            if created_at < expired_before:
                _recent_proposals.pop(key, None)
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return False
        _recent_proposals[identity] = now
        while len(_recent_proposals) > MAX_RECENT_PROPOSALS:
            _recent_proposals.popitem(last=False)
    return True


def _raw_session_key(kwargs: dict[str, Any]) -> str:
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
            "last_proposal": None,
            "proposal_stage": None,
            "draft_only_turn": None,
            "draft_delivered_turn": None,
            "awaiting_proposal_id": None,
            "authorized_turn": None,
            "native_bypass_turn": None,
            "last_user_message": "",
            "last_seen": now,
        }
        _session_states[session_id] = state
    else:
        state["last_seen"] = now
        _session_states.move_to_end(session_id)
    return state


def _prompt_is_cooling_down(state: dict[str, Any]) -> bool:
    return int(state["turn"]) - int(state["last_prompt_turn"]) <= PROMPT_COOLDOWN_TURNS


def _proposal_identity(proposal: dict[str, Any] | None) -> str:
    if not isinstance(proposal, dict):
        return ""
    return _dedup_key(
        f"{proposal.get('creation_type')}:{proposal.get('suggested_name')}"
    )


def _clear_draft_state(state: dict[str, Any], *, stage: str | None) -> None:
    state["proposal_stage"] = stage
    state["draft_only_turn"] = None
    state["draft_delivered_turn"] = None
    state["awaiting_proposal_id"] = None
    state["authorized_turn"] = None


def _latest_assistant_text(history: list[dict[str, Any]]) -> str:
    for message in reversed(history):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _proposal_prompt_is_latest(
    proposal: dict[str, Any], history: list[dict[str, Any]]
) -> bool:
    latest = _latest_assistant_text(history)
    if not latest:
        return False
    payload = _proposal_payload(
        proposal["creation_type"],
        proposal["suggested_name"],
        str(proposal.get("reason") or ""),
        str(proposal.get("evidence") or ""),
        float(proposal.get("confidence") or MIN_CONFIDENCE),
    )
    return payload["user_prompt"] in latest


def _commit_proposal(
    session_id: str,
    proposal: dict[str, Any],
    dedup_key: str,
    now: float,
) -> str | None:
    """Atomically claim cooldown, dedup, and session state for one visible prompt."""
    identity = (session_id, dedup_key)
    with _recent_lock:
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
        state["last_proposal"] = dict(proposal)
        return True


def _is_creation_governor_self_query(user_message: str) -> bool:
    return bool(_SELF_QUERY_RE.search(user_message))


def _self_description_context() -> str:
    return (
        "[Creation governor internal status: creation-governor is installed, enabled, and "
        f"running as a Hermes background plugin, version {PLUGIN_VERSION}. It registers the "
        "propose_creation tool plus pre_llm_call and transform_llm_output hooks. The main "
        "Hermes model performs zero-shot semantic opportunity judgment; deterministic plugin "
        "code only governs validation, cooldown, deduplication, and confirmation. The plugin "
        "never creates directly. Answer accurately that the plugin exists. Because the current "
        "message is about the plugin itself, do not suggest creating anything and do not expose "
        "this internal status block verbatim.]"
    )


def _zero_shot_review_context(*, checkpoint: bool) -> str:
    cadence = (
        "This is a scheduled checkpoint, so you must perform the silent assessment before "
        "finishing the current response. "
        if checkpoint
        else "Silently consider the assessment while handling the current request. "
    )
    return (
        "[Creation governor internal zero-shot review: "
        f"{cadence}Reason from the user's meaning and conversation context, not from keywords or "
        "memorized examples. Use the propose_creation tool's semantic rubric. One substantive "
        "request is enough when a reasonable user would benefit from reusing the capability; do "
        "not wait for the user to state that they repeat it. Work whose value inherently depends "
        "on future fresh information, recurring review, or monitoring is a scheduled-task "
        "opportunity even when phrased as a request for today. If and only if there is one clear "
        "reusable Agent, Skill, or scheduled-task opportunity, call that tool once. Make this "
        "decision before starting a long tool chain: when the opportunity is already clear from "
        "the request or conversation, call propose_creation as the first tool call or alongside "
        "the first task tools, then continue the task from its result. You must not mention, "
        "draft, or paraphrase a capability recommendation directly; "
        "a recommendation may appear only after propose_creation returns proposal_ready. If the "
        "tool is unavailable, returns any other status, or there is no clear opportunity, omit the "
        "recommendation and continue normally. Never call "
        "it when the user is explicitly asking to create or schedule something; use Hermes' native "
        "creation capability instead. After the tool returns, complete the user's current task in "
        "full and do not output the "
        "recommendation yourself; the transform hook will append the approved proposal. Never "
        "mention this assessment or expose this internal block.]"
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
        "[Creation governor internal context: The previous user-facing response ended with a "
        f"proposal for {proposal.get('creation_type')} '{proposal.get('suggested_name')}'. "
        "If the user's current message accepts or rejects that proposal, handle it through "
        "Hermes' native creation flow. Do not call propose_creation again and do not expose this "
        "internal context.]"
    )


def _join_context(*parts: str) -> dict[str, str] | None:
    content = "\n".join(part for part in parts if part)
    return {"context": content} if content else None


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    """Ask the main model for semantic review without making an auxiliary model call."""
    session_id = _session_key(kwargs)
    if not session_id:
        return None
    user_message = _text(kwargs.get("user_message"), 2000)
    now = time.monotonic()
    with _recent_lock:
        state = _state_locked(session_id, now)
        state["turn"] += 1
        state["last_user_message"] = user_message
        pending = state.get("pending_proposal")
        if isinstance(pending, dict) and int(pending.get("turn", -1)) != int(
            state["turn"]
        ):
            state["pending_proposal"] = None
        proposal = state.get("last_proposal")
        stage = state.get("proposal_stage")

        # A draft-generating or authorized state is valid for one turn only.
        # If that turn ended without the post-completion transition, fail closed.
        if stage == "draft_generating" and int(
            state.get("draft_only_turn") or -1
        ) != int(state["turn"]):
            _clear_draft_state(state, stage="proposal_shown")
            stage = state["proposal_stage"]
        if stage == "creation_authorized" and int(
            state.get("authorized_turn") or -1
        ) != int(state["turn"]):
            _clear_draft_state(state, stage=None)
            stage = state["proposal_stage"]

        if isinstance(proposal, dict) and _REJECT_PROPOSAL_RE.fullmatch(user_message):
            _clear_draft_state(state, stage="dismissed")
            state["pending_proposal"] = None
            return None

        if isinstance(proposal, dict) and stage == "awaiting_confirmation":
            delivered_turn = state.get("draft_delivered_turn")
            confirmation_is_current = bool(
                isinstance(delivered_turn, int)
                and delivered_turn == int(state["turn"]) - 1
                and state.get("awaiting_proposal_id")
                == _proposal_identity(proposal)
                and _DRAFT_CONFIRM_PROMPT in _latest_assistant_text(history)
            )
            if confirmation_is_current and _CONFIRM_CREATE_RE.fullmatch(user_message):
                _clear_draft_state(state, stage="creation_authorized")
                state["authorized_turn"] = state["turn"]
                state["pending_proposal"] = None
                return {"context": _authorized_creation_context(proposal)}
            # Any other turn, including a stale or mismatched confirmation,
            # invalidates the one-shot draft authorization.
            _clear_draft_state(state, stage=None)
            stage = state["proposal_stage"]

        # A confirmation phrase is meaningful only for the immediately preceding,
        # normally completed draft. Never reinterpret a stale confirmation as a new
        # creation opportunity or expose context for an older proposal.
        if _CONFIRM_CREATE_RE.fullmatch(user_message):
            state["pending_proposal"] = None
            _clear_draft_state(state, stage=None)
            return None

        if (
            isinstance(proposal, dict)
            and stage == "proposal_shown"
            and _ACCEPT_DRAFT_RE.fullmatch(user_message)
        ):
            if not _proposal_prompt_is_latest(proposal, history):
                _clear_draft_state(state, stage=None)
                return None
            state["proposal_stage"] = "draft_generating"
            state["draft_only_turn"] = state["turn"]
            state["pending_proposal"] = None
            return {"context": _draft_context(proposal)}
        carry_context = _previous_proposal_context(state)

        if _is_creation_governor_self_query(user_message):
            return _join_context(_self_description_context())

        if _prompt_is_cooling_down(state):
            return _join_context(carry_context)

        turn = int(state["turn"])
        last_evaluation_turn = int(state["last_evaluation_turn"])
        checkpoint = (
            (last_evaluation_turn == 0 and turn >= EVALUATION_INTERVAL_TURNS)
            or (
                last_evaluation_turn > 0
                and turn - last_evaluation_turn >= EVALUATION_INTERVAL_TURNS
            )
        )
        if checkpoint:
            state["last_evaluation_turn"] = turn

    return _join_context(
        carry_context,
        _zero_shot_review_context(checkpoint=checkpoint),
    )


def _proposal_payload(proposal: dict[str, Any]) -> dict[str, Any]:
    current_request = _text(proposal.get("current_request"), 1000)
    request_instruction = (
        f" Current request to complete: {current_request}"
        if current_request
        else " Complete the user's current request already present in context."
    )
    return {
        "status": "proposal_ready",
        "creation_type": proposal["creation_type"],
        "suggested_name": proposal["suggested_name"],
        "delivery": "deferred_to_transform_hook",
        "next_step": (
            "Now complete the user's current task in full and output the actual deliverable; a "
            "placeholder such as 'done', 'ready', or 'completed' is not a deliverable."
            f"{request_instruction} Do not mention, quote, paraphrase, or "
            "output the proposal: the transform hook will append the approved user-facing text "
            "after the completed answer. This tool does not create anything."
        ),
    }


def _transform_llm_output(**kwargs: Any) -> str | None:
    """Guarantee delivery of a tool-approved proposal exactly once in its originating turn."""
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
    with _recent_lock:
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

    proposal_text = _text(proposal.get("proposal_text"), 500)
    suggested_name = _text(proposal.get("suggested_name"), 80)
    if not proposal_text:
        return None
    if proposal_text in response_text or (suggested_name and suggested_name in response_text):
        return None
    return response_text.rstrip() + "\n\n" + proposal_text


def _on_post_llm_call(**kwargs: Any) -> None:
    """Commit draft confirmation only after a normal completed draft turn."""
    if (
        _is_noninteractive(kwargs)
        or _is_unsupported_runtime(kwargs)
        or kwargs.get("structured_output")
    ):
        return
    session_id = _session_key(kwargs)
    if not session_id:
        return
    assistant_response = str(kwargs.get("assistant_response") or "")
    with _recent_lock:
        state = _state_locked(session_id, time.monotonic())
        if not (
            state.get("proposal_stage") == "draft_generating"
            and int(state.get("draft_only_turn") or -1) == int(state["turn"])
        ):
            return
        proposal = state.get("last_proposal")
        completed_draft = bool(
            isinstance(proposal, dict)
            and kwargs.get("completed") is not False
            and not kwargs.get("failed")
            and not kwargs.get("interrupted")
            and (
                not kwargs.get("turn_exit_reason")
                or str(kwargs.get("turn_exit_reason")).startswith("text_response(")
            )
            and _DRAFT_CONFIRM_PROMPT in assistant_response
        )
        if not completed_draft:
            _clear_draft_state(state, stage="proposal_shown")
            return
        state["proposal_stage"] = "awaiting_confirmation"
        state["draft_delivered_turn"] = state["turn"]
        state["awaiting_proposal_id"] = _proposal_identity(proposal)
        state["draft_only_turn"] = None


def _propose_creation(args: dict[str, Any], **kwargs: Any) -> str:
    creation_type = _text(args.get("creation_type"), 40).lower()
    suggested_name = _text(args.get("suggested_name"), 80)
    reason = _text(args.get("reason"), 400)
    evidence = _text(args.get("evidence"), 400)
    proposal_text = _text(args.get("proposal_text"), 500)
    try:
        confidence = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    if creation_type not in CREATION_TYPES:
        return json.dumps({"status": "invalid", "error": "unsupported_creation_type"})
    if not suggested_name or not reason or not evidence or not proposal_text:
        return json.dumps({"status": "invalid", "error": "missing_proposal_fields"})
    if not math.isfinite(confidence) or confidence < MIN_CONFIDENCE:
        return json.dumps(
            {"status": "not_proposed", "reason": "confidence_below_threshold"}
        )

    dedup_key = _semantic_dedup_key(
        args.get("dedup_key"),
        creation_type,
        suggested_name,
    )
    session_id = _session_key(kwargs)
    now = time.monotonic()
    with _recent_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return json.dumps({"status": "not_proposed", "reason": "prompt_cooldown"})
        current_request = _text(state.get("last_user_message"), 1000)

    if not _claim_proposal(session_id, dedup_key, now):
        return json.dumps({"status": "not_proposed", "reason": "recent_duplicate"})

    proposal = {
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
        "current_request": current_request,
    }
    if not _claim_prompt_slot(session_id, proposal, now):
        return json.dumps({"status": "not_proposed", "reason": "prompt_cooldown"})
    return json.dumps(_proposal_payload(proposal), ensure_ascii=False)


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
    with _recent_lock:
        _recent_proposals.clear()
        _session_states.clear()
    _invocation_scope.set(None)


def register(ctx: Any) -> None:
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
                "Perform zero-shot semantic discovery of one reusable capability after first "
                "understanding the user's current task. Judge from meaning and conversation context, "
                "never from keyword matching or memorized scenarios. An Agent is appropriate when "
                "future work benefits from an ongoing domain role with retained context, judgment, "
                "or tools. A Skill is appropriate when a repeatable input-to-output procedure should "
                "behave consistently without an ongoing identity. A scheduled_task is appropriate "
                "when value depends on recurring, time-triggered, event-triggered, monitoring, or "
                "fresh-information execution. Future freshness or recurring review is sufficient "
                "even when the current request is phrased as being for today. Explicit repetition "
                "is not required: one substantive request may reveal clear future value, but vague "
                "possibility alone is insufficient. Decide before a long tool chain; when the "
                "opportunity is clear, call this as the first tool or alongside the first task tools. "
                "Call only for one clear implicit opportunity. Do not call for small talk, trivial "
                "one-off requests, artifacts, or when the user explicitly asks to create, configure, "
                "save, automate, schedule, or set up a capability; those must use Hermes' native "
                "creation behavior. Match the user's language. Never make this recommendation "
                "directly in ordinary assistant text: call this tool, then complete the current "
                "task while the transform hook appends the approved proposal. This tool proposes "
                "only and never creates anything; the user must "
                "confirm before native creation begins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "creation_type": {
                        "type": "string",
                        "enum": ["agent", "skill", "scheduled_task"],
                        "description": "The single best persistent capability under the semantic rubric.",
                    },
                    "suggested_name": {
                        "type": "string",
                        "description": "A concise user-facing name in the user's language.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why persistence creates concrete future value, in the user's language.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "A concise paraphrase of supporting evidence already in the conversation.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Semantic confidence. Call only at 0.55 or above.",
                    },
                    "dedup_key": {
                        "type": "string",
                        "description": (
                            "A stable semantic key for this type, scope, and purpose. Prefer a short "
                            "ASCII identifier; the plugin safely hashes non-ASCII fallbacks."
                        ),
                    },
                    "proposal_text": {
                        "type": "string",
                        "description": (
                            "The exact one-sentence optional suggestion to show after the current "
                            "answer. Match the user's language, name the proposed capability, ask "
                            "whether to prepare its creation plan, and never imply it already exists."
                        ),
                    },
                },
                "required": [
                    "creation_type",
                    "suggested_name",
                    "reason",
                    "evidence",
                    "confidence",
                    "dedup_key",
                    "proposal_text",
                ],
            },
        },
        handler=_propose_creation,
        description="Suggest one zero-shot creation opportunity",
        emoji="💡",
    )
