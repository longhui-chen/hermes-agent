"""Per-user-turn budget for rebuilding / republishing one generated app.

Why this module exists
----------------------
The ``app-builder`` skill has always *told* the model to stop after about three
failed repair rounds and hand the situation back to the user. Nothing ever
counted. A single board turn was observed running three hours and 431 tool
calls, republishing the same app over and over: the model never reached the
"stop and report" state the prompt describes, because no code ever said no.
This module is that code. The prompt text and this file must agree — the skill
now states the limit is platform-enforced, not a convention.

Where the gate sits, and why here
---------------------------------
In hermes, not in local-server. Writing source, compiling, self-testing and
health-checking all happen through ``terminal`` inside the app's own directory,
so local-server only ever witnesses the tail of a round (``acquire_slot`` /
``publish``) and cannot tell a first delivery from a tenth repair attempt. Its
build slot is an in-memory lease that counts nothing, and its operation journal
only persists for publications that carry a maintenance block. Hermes'
``tools.apphost_tool.app_host_tool`` is the single funnel every credentialed
App Host action passes through, so the count is taken there and the refusal is
produced *before* any HTTP request leaves the tool.

What "one repair task" means
----------------------------
One user turn. The user speaking is what clears the counters, because a turn
boundary is the only moment a human has had the chance to say "keep going".
The turn identity comes from :func:`gateway.session_context.current_turn_identity`,
whose opaque binding object is minted fresh per request and cannot be
reconstructed from model- or client-supplied metadata.

Context compression deliberately does NOT reset anything. Compression happens
*inside* a turn and keeps the same task context, so the same binding object is
still current; a model that has just been compressed is exactly the one most
likely to have forgotten how many rounds it already burned, so resetting there
would remove the guardrail precisely when it is needed.

Counting rules that are not obvious
-----------------------------------
``acquire_slot`` counts only when a slot is actually GRANTED. The server
answers a queued caller immediately with ``{queue_ahead}`` and the skill polls
every ten seconds until a ``token`` appears, so counting calls would burn the
whole budget inside a minute of queueing. A granted slot is one compile round.

``publish`` / ``install`` / ``reload`` count whenever the request actually left
the tool, success or failure alike: a failed publish still consumed a round of
the user's time and of the device's CPU, and "it failed so it does not count"
is exactly the reasoning that produced the three-hour turn. Requests the tool
rejected locally (``status == 0``, nothing was sent) do not count — a malformed
argument is not a repair round.

This module never raises. Every public entry point swallows its own errors and
degrades to "no opinion": a bookkeeping bug must not break the tool call the
model is waiting on, and the per-turn loop cap in ``agent.tool_guardrails``
remains as the backstop.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Mapping


# ── Budgets ──────────────────────────────────────────────────────────────
# One delivery plus three repair rounds — the count the skill has always
# described in prose. 0 disables a budget.
MAX_BUILD_ROUNDS_PER_TURN = 4
MAX_PUBLISHES_PER_APP_PER_TURN = 4

# App Host actions that submit a new version of an app. install/reload are the
# legacy staging channel for the same thing and must not be a way around the
# count.
PUBLISH_ACTIONS = frozenset({"publish", "install", "reload"})
# One granted build slot == one compile round.
BUILD_ACTIONS = frozenset({"acquire_slot"})

PUBLISH_BUDGET_CODE = "publish_budget_exhausted"
BUILD_BUDGET_CODE = "build_budget_exhausted"

# Bucket for a publish whose app cannot be named from the arguments (a legacy
# source_subdir publish with no slug). Sharing one bucket is deliberately
# stricter than minting a bucket per call.
_UNSCOPED = "*"

_MAX_TRACKED_TURNS = 16
_MAX_TRACKED_APPS = 32
_MAX_SITE_CHARS = 512

_lock = threading.RLock()
_ledgers: "OrderedDict[Any, _TurnLedger]" = OrderedDict()
# Fallback turn counter for entry points that do not bind a turn identity (the
# CLI and the single-profile daemon). Advanced by :func:`note_turn_boundary`
# from ``agent.turn_context.reset_for_turn``, the one per-user-turn reset point.
_local_turn_generation = 0


class _TurnLedger:
    """What one user turn has already spent on app changes."""

    __slots__ = ("build_rounds", "publishes", "sites")

    def __init__(self) -> None:
        self.build_rounds = 0
        self.publishes: dict[str, int] = {}
        # slug -> the directory `prepare` handed back, so a refusal can name
        # the live site instead of telling the model to remember it.
        self.sites: dict[str, str] = {}


def note_turn_boundary() -> None:
    """Advance the fallback turn generation (a new user turn has started).

    A no-op for requests that carry a real turn identity — those key off the
    identity itself, which already changes per turn.
    """
    global _local_turn_generation
    try:
        with _lock:
            _local_turn_generation += 1
    except Exception:
        pass


def refuse_if_exhausted(args: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """Return ``(error_code, message)`` when this turn's budget is spent.

    ``None`` means "carry on". The caller must turn a returned pair into a
    local failure envelope WITHOUT sending the request.
    """
    try:
        if not isinstance(args, Mapping):
            return None
        action = _clean(args.get("action"))
        if action in BUILD_ACTIONS:
            cap = MAX_BUILD_ROUNDS_PER_TURN
            if not cap:
                return None
            ledger = _ledger(create=False)
            used = ledger.build_rounds if ledger is not None else 0
            if used < cap:
                return None
            return BUILD_BUDGET_CODE, _build_refusal(used, ledger)
        if action in PUBLISH_ACTIONS:
            cap = MAX_PUBLISHES_PER_APP_PER_TURN
            if not cap:
                return None
            scope = app_scope(args)
            ledger = _ledger(create=False)
            used = ledger.publishes.get(scope or _UNSCOPED, 0) if ledger is not None else 0
            if used < cap:
                return None
            return PUBLISH_BUDGET_CODE, _publish_refusal(scope, used, ledger)
        return None
    except Exception:
        return None


def observe(args: Mapping[str, Any] | None, result_json: Any) -> None:
    """Book one completed App Host call against the current turn."""
    try:
        if not isinstance(args, Mapping):
            return
        action = _clean(args.get("action"))
        if action not in PUBLISH_ACTIONS and action not in BUILD_ACTIONS and action != "prepare":
            return
        if not isinstance(result_json, str):
            return
        try:
            parsed = json.loads(result_json)
        except (TypeError, ValueError):
            return
        if not isinstance(parsed, dict):
            return
        ok = parsed.get("ok") is True
        data = parsed.get("data")
        if not isinstance(data, dict):
            data = {}

        if action == "prepare":
            if not ok:
                return
            directory = _clean(data.get("dir"))[:_MAX_SITE_CHARS]
            slug = app_scope(args)
            if directory and slug:
                ledger = _ledger(create=True)
                if ledger is not None:
                    ledger.sites[slug] = directory
                    _trim(ledger.sites)
            return

        if action in BUILD_ACTIONS:
            # Queued callers get {queue_ahead} and poll; only a token is a round.
            if not (ok and _clean(data.get("token"))):
                return
            ledger = _ledger(create=True)
            if ledger is not None:
                ledger.build_rounds += 1
            return

        # publish / install / reload
        if not _request_left_the_tool(parsed, ok):
            return
        ledger = _ledger(create=True)
        if ledger is not None:
            scope = app_scope(args) or _UNSCOPED
            ledger.publishes[scope] = ledger.publishes.get(scope, 0) + 1
            _trim(ledger.publishes)
    except Exception:
        return


def app_scope(args: Mapping[str, Any] | None) -> str:
    """Name the app a call acts on, or ``""`` when the arguments do not say."""
    if not isinstance(args, Mapping):
        return ""
    slug = _clean(args.get("slug"))
    if slug:
        return slug[:128]
    # Legacy staging publishes can carry a directory instead of a slug.
    for key in ("source_subdir", "staging_dir", "app_path"):
        value = _clean(args.get(key))
        if value:
            tail = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return (tail or value)[:128]
    return ""


def usage_summary(slug: str = "") -> str:
    """One human line describing what this turn already spent, for a halt message."""
    try:
        ledger = _ledger(create=False)
        if ledger is None:
            return "本回合尚无平台记录"
        name = _clean(slug)
        if name and name in ledger.publishes:
            published = ledger.publishes[name]
        else:
            published = sum(ledger.publishes.values())
        parts = [f"本回合已提交新版本 {published} 次", f"编译 {ledger.build_rounds} 轮"]
        site = site_hint(name) if name else ""
        if not site and len(ledger.sites) == 1:
            site = next(iter(ledger.sites.values()))
        if site:
            parts.append(f"现场在 {site}")
        return "；".join(parts)
    except Exception:
        return "本回合尚无平台记录"


def site_hint(slug: str) -> str:
    """The directory ``prepare`` handed back for ``slug`` this turn, if known."""
    try:
        ledger = _ledger(create=False)
        if ledger is None:
            return ""
        return ledger.sites.get(_clean(slug), "")
    except Exception:
        return ""


def reset_all_for_tests() -> None:
    """Drop every ledger. Test-only; production clears by turn identity."""
    global _local_turn_generation
    with _lock:
        _ledgers.clear()
        _local_turn_generation += 1


# ── internals ────────────────────────────────────────────────────────────


def _turn_key() -> Any:
    """The key a budget belongs to: the trusted turn binding when there is one."""
    try:
        from gateway.session_context import current_turn_identity

        identity = current_turn_identity()
    except Exception:
        identity = None
    if identity is not None:
        # (turn_id, opaque binding). The binding is minted per request by
        # set_turn_vars, survives compression (same task context) and cannot be
        # forged from client metadata. Never serialize it.
        return identity
    with _lock:
        return ("hermes-local-turn", _local_turn_generation)


def _ledger(*, create: bool) -> _TurnLedger | None:
    key = _turn_key()
    with _lock:
        ledger = _ledgers.get(key)
        if ledger is None:
            if not create:
                return None
            ledger = _TurnLedger()
            _ledgers[key] = ledger
            while len(_ledgers) > _MAX_TRACKED_TURNS:
                _ledgers.popitem(last=False)
        else:
            _ledgers.move_to_end(key)
        return ledger


def _request_left_the_tool(parsed: Mapping[str, Any], ok: bool) -> bool:
    """Did this call actually reach App Host?

    ``apphost_tool`` publishes a three-way ``status``: an HTTP code (the server
    answered), ``None`` (it went out, outcome unknown) and ``0`` (rejected
    locally, not one byte sent). Only the last one is free.
    """
    if ok:
        return True
    return parsed.get("status") != 0


def _trim(mapping: dict) -> None:
    while len(mapping) > _MAX_TRACKED_APPS:
        mapping.pop(next(iter(mapping)), None)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _site_clause(slug: str, ledger: _TurnLedger | None) -> str:
    site = ""
    if ledger is not None:
        site = ledger.sites.get(slug, "")
        if not site and len(ledger.sites) == 1:
            site = next(iter(ledger.sites.values()))
    if site:
        return f"现场留在哪：应用目录 {site}，原样留着没动"
    return "现场留在哪：prepare 返回的那个应用目录，把完整路径照实说出来，它原样留着没动"


def _tail() -> str:
    return (
        "然后就结束本回合。不要再改代码、不要再编译、不要再发布，"
        "也不许自行回滚或删除——撤销和删除只能由用户发起。\n"
        "这是平台按「一个用户回合」强制执行的上限，不是让你自觉遵守的约定；"
        "用户下一句话会把它清零。"
    )


def _publish_refusal(slug: str, used: int, ledger: _TurnLedger | None) -> str:
    who = f"「{slug}」" if slug else "这个应用"
    return (
        f"本回合已经给{who}提交过 {used} 次新版本，问题仍未解决。"
        "这次发布没有发出去——平台在本地就拒绝了，App Host 一个字节都没收到。\n"
        "现在停下来，把下面四件事讲给用户：\n"
        "1. 卡在哪：最后一次失败的真实报错，原文别改写、别概括成「还有点问题」；\n"
        f"2. 这 {used} 轮各改了什么、各自的结果是什么；\n"
        f"3. {_site_clause(slug, ledger)}；\n"
        "4. 你的建议：接着修，还是先放弃。\n"
        + _tail()
    )


def _build_refusal(used: int, ledger: _TurnLedger | None) -> str:
    site = ""
    if ledger is not None and len(ledger.sites) == 1:
        site = next(iter(ledger.sites.values()))
    site_clause = (
        f"现场留在哪：应用目录 {site}，原样留着没动"
        if site
        else "现场留在哪：prepare 返回的那个应用目录，把完整路径照实说出来，它原样留着没动"
    )
    return (
        f"本回合已经编译过 {used} 轮，问题仍未解决。"
        "这次编译槽申请没有发出去——平台在本地就拒绝了，App Host 一个字节都没收到。\n"
        "现在停下来，把下面四件事讲给用户：\n"
        "1. 卡在哪：最后一次编译失败的真实报错，原文别改写；\n"
        f"2. 这 {used} 轮各改了什么、各自的结果是什么；\n"
        f"3. {site_clause}；\n"
        "4. 你的建议：接着修，还是先放弃。\n"
        + _tail()
    )
