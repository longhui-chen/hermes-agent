"""Search Memory Tool — unified read-only recall over persistent memory.

Two-layer retrieval:
  1. Built-in curated memory searches MEMORY.md and USER.md entries.
  2. An activated provider may expose ``search(query, top_k)`` plus a
     ``search_memory_mode()`` policy. ``replace`` preserves the legacy
     provider-first contract, ``supplement`` combines both sources, and
     ``disabled`` leaves native recall untouched.

Result contract (JSON string):
    {"items": [{"id", "source", "excerpt", "score"}, ...],
     "provider"?: str, "provider_status"?: str}

``id`` is a short content hash (sha1, first 12 hex chars) of the full entry —
stable across sessions so later memory.citations work can reference it.

HARD CONSTRAINT: recall failure must never block the answer. Every path —
missing files, malformed provider results, provider exceptions — degrades to
``{"items": []}``; this tool never raises.
"""

import hashlib
import json
import logging
import re
import threading

logger = logging.getLogger(__name__)

_EXCERPT_MAX_CHARS = 200
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 25
# 🔴 provider.search() 是**别人实现的**扩展点，可能打网络、可能问守护进程。
# 直接在 agent 线程上同步调它 ⇒ 它一卡，search_memory 工具和复用它的 turn 收尾
# 归因就把整轮钉死在 running；上面那句 `except Exception` 的降级**永远不会触发**，
# 因为「卡住」不是异常。需求 3.2 明写「召回失败不得阻塞回答」，超时同理。
# ⇒ 与 memory_manager._prefetch_provider 同款：有界 worker + join 超时，逾时按
#   「这个 provider 没结果」处理、落回内置策展记忆。⚠️ 线程是 daemon，卡死的调用
#   不会拖住进程退出。
_PROVIDER_SEARCH_TIMEOUT_S = 8.0

# ASCII words/numbers as whole tokens; CJK ideographs (U+3400-U+4DBF,
# U+4E00-U+9FFF) as single-char tokens — curated memory is Chinese-heavy and
# has no word boundaries to split on.
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[㐀-䶿一-鿿]")

SEARCH_MEMORY_SCHEMA = {
    "name": "search_memory",
    "description": (
        "Search your persistent memory (personal notes + user profile, and an "
        "external memory provider when one is active) for entries relevant to "
        "a query. Read-only. Use to recall stored facts, preferences, and "
        "conventions relevant to the current task before asking the user to "
        "repeat themselves. Returns the top matching entries with a stable id, "
        "source, excerpt, and relevance score; an empty item list means "
        "nothing relevant is stored — never an error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for (keywords or a short phrase).",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of entries to return (default 5).",
            },
        },
        "required": ["query"],
    },
}


def _entry_id(entry: str) -> str:
    """Stable short id for a curated entry: sha1 of the FULL entry content,
    first 12 hex chars. Later memory.citations work relies on this being
    derived from content (not position), so it survives reordering."""
    return hashlib.sha1(entry.encode("utf-8")).hexdigest()[:12]


def _excerpt(entry: str) -> str:
    entry = entry.strip()
    if len(entry) <= _EXCERPT_MAX_CHARS:
        return entry
    return entry[:_EXCERPT_MAX_CHARS] + "…"


def _coerce_top_k(raw) -> int:
    try:
        top_k = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TOP_K
    return max(1, min(top_k, _MAX_TOP_K))


def _read_curated_entries():
    """Yield (source, entry) pairs from the profile-scoped curated memory files.

    Bounded read (memory files are capped at MAX_CURATED_MEMORY_FILE_BYTES by
    the memory tool; we enforce the same ceiling defensively). Missing or
    unreadable files simply contribute no entries.
    """
    # Import the module (not the symbols) so tests monkeypatching
    # tools.memory_tool.get_memory_dir are honored, matching the existing
    # memory-tool test fixtures.
    from tools import memory_tool as _memory_tool

    pairs = []
    try:
        mem_dir = _memory_tool.get_memory_dir()
    except Exception:
        return pairs
    for source, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
        try:
            with open(mem_dir / filename, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read(_memory_tool.MAX_CURATED_MEMORY_FILE_BYTES)
        except OSError:
            continue
        # Split on the full delimiter, not bare "§" — entries may legitimately
        # contain "§" in their content (same rationale as MemoryStore).
        for entry in raw.split(_memory_tool.ENTRY_DELIMITER):
            entry = entry.strip()
            if entry:
                pairs.append((source, entry))
    return pairs


def _score_entry(entry_lower: str, tokens, phrase: str) -> float:
    """Token-level relevance: one point per distinct query token found in the
    entry, plus a full-phrase bonus (worth as much as all tokens together)
    when the whole normalized query appears verbatim."""
    hits = sum(1 for tok in tokens if tok in entry_lower)
    if hits == 0:
        return 0.0
    score = float(hits)
    if phrase and phrase in entry_lower:
        score += float(len(tokens))
    return score


def _search_curated(query: str, top_k: int):
    phrase = re.sub(r"\s+", " ", query.strip().lower())
    tokens = list(dict.fromkeys(_TOKEN_RE.findall(phrase)))
    if not tokens:
        return []
    scored = []
    for source, entry in _read_curated_entries():
        score = _score_entry(entry.lower(), tokens, phrase)
        if score > 0:
            scored.append({
                "id": _entry_id(entry),
                "source": source,
                "excerpt": _excerpt(entry),
                "score": score,
            })
    # sorted() is stable: ties keep file order (MEMORY.md before USER.md).
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _normalize_provider_items(raw, provider_name: str, top_k: int):
    """Coerce a provider's search() result into the tool's item contract.

    Accepts a list (of dicts or strings) or a {"items": [...]} dict. Anything
    else normalizes to no items — a misbehaving provider must not block."""
    if isinstance(raw, dict):
        raw = raw.get("items")
    if not isinstance(raw, list):
        return []
    items = []
    for element in raw[:top_k]:
        if isinstance(element, dict):
            text = str(
                element.get("excerpt")
                or element.get("text")
                or element.get("content")
                or element.get("statement")
                or element.get("fact")
                or ""
            )
            try:
                score = float(
                    element.get(
                        "score",
                        element.get(
                            "relevance_score",
                            element.get("recall_score", element.get("relevance", 0.0)),
                        ),
                    )
                )
            except (TypeError, ValueError):
                score = 0.0
            items.append({
                "id": str(
                    element.get("id")
                    or element.get("memory_id")
                    or _entry_id(text)
                ),
                "source": str(element.get("source") or provider_name),
                "excerpt": _excerpt(text),
                "score": score,
            })
        elif isinstance(element, str):
            items.append({
                "id": _entry_id(element),
                "source": provider_name,
                "excerpt": _excerpt(element),
                "score": 0.0,
            })
    return items


def _find_provider_search(memory_manager):
    """Return the first active provider's name, search callable, and policy."""
    if memory_manager is None:
        return None
    try:
        providers = list(getattr(memory_manager, "providers", []) or [])
    except Exception:
        return None
    for provider in providers:
        search_fn = getattr(provider, "search", None)
        if callable(search_fn):
            mode = "replace"
            mode_fn = getattr(provider, "search_memory_mode", None)
            if callable(mode_fn):
                try:
                    mode = str(mode_fn() or "replace").strip().lower()
                except Exception:
                    logger.warning(
                        "memory provider search policy failed; disabling its search",
                        exc_info=True,
                    )
                    continue
            if mode == "disabled":
                continue
            if mode not in {"replace", "supplement"}:
                logger.warning(
                    "memory provider returned unsupported search policy %r; disabling its search",
                    mode,
                )
                continue
            try:
                name = str(getattr(provider, "name", "") or "")
            except Exception:
                name = ""
            return (name or type(provider).__name__, search_fn, mode)
    return None


def _provider_status(raw) -> str:
    if isinstance(raw, dict):
        status = str(raw.get("status") or "").strip().lower()
        if status:
            return status[:64]
        items = raw.get("items")
        return "ok" if isinstance(items, list) and items else "empty"
    return "ok" if isinstance(raw, list) and raw else "empty"


def _failure_status(failure) -> str:
    if isinstance(failure, TimeoutError) or "timed out" in str(failure).lower():
        return "timeout"
    return "unavailable"


def _merge_supplement_items(curated, provider_items, top_k):
    """Round-robin native-first so both ranked sources can contribute.

    Scores from local token matching and external retrieval are not directly
    comparable. Interleaving preserves each source's own ranking, keeps native
    memory primary, and still gives a supplemental provider useful slots.
    """
    merged = []
    seen_ids = set()
    seen_text = set()
    width = max(len(curated), len(provider_items))
    for index in range(width):
        for items in (curated, provider_items):
            if index >= len(items):
                continue
            item = items[index]
            item_id = str(item.get("id") or "").strip()
            text_key = re.sub(r"\s+", " ", str(item.get("excerpt") or "").strip().lower())
            if (item_id and item_id in seen_ids) or (text_key and text_key in seen_text):
                continue
            merged.append(item)
            if item_id:
                seen_ids.add(item_id)
            if text_key:
                seen_text.add(text_key)
            if len(merged) >= top_k:
                return merged
    return merged


def _call_provider_search_bounded(provider_name, search_fn, query, top_k):
    """跑 provider.search()，带超时。

    返回 ``(raw, failure)``：``failure is None`` 才代表拿到了可用结果；否则
    ``failure`` 是一句可记日志的原因（异常或超时），调用方据此降级。
    ⛔ 不要把超时并进 ``except``——超时不是异常，同步调用卡住时那条路根本不走。
    """
    box = {}

    def _run():
        try:
            box["value"] = search_fn(query, top_k)
        except Exception as exc:  # noqa: BLE001 - 交给调用方统一降级
            box["error"] = exc

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name=f"memory-search-{provider_name}",
    )
    thread.start()
    thread.join(_PROVIDER_SEARCH_TIMEOUT_S)
    if thread.is_alive():
        # ⚠️ 线程留着继续跑（daemon，进程退出即死）；本轮按「没结果」处理，
        # 不等它，也不试图杀它——Python 没有安全的线程中断。
        return None, f"timed out after {_PROVIDER_SEARCH_TIMEOUT_S:.1f}s"
    if "error" in box:
        return None, box["error"]
    return box.get("value"), None


def search_memory_tool(args, **kw):
    # Tool handlers must return a STRING (json-encoded) — same contract as
    # memory/list_my_channels. NEVER raise: recall failure must not block the
    # answer, so every failure path degrades to {"items": []}.
    try:
        args = args or {}
        query = str(args.get("query") or "").strip()
        top_k = _coerce_top_k(args.get("top_k", _DEFAULT_TOP_K))
        if not query:
            return json.dumps({"items": []}, ensure_ascii=False)

        # A provider can replace or supplement the native curated layer.
        # ``memory_manager`` is threaded in by the agent runtime
        # (agent/agent_runtime_helpers.py), mirroring how the memory tool
        # receives its store; registry-only dispatch paths fall through to
        # the curated layer.
        found = (
            _find_provider_search(kw.get("memory_manager"))
            if kw.get("supplement_external", True)
            else None
        )
        if found is not None:
            provider_name, search_fn, mode = found
            raw, failure = _call_provider_search_bounded(provider_name, search_fn, query, top_k)
            if failure is None:
                items = _normalize_provider_items(raw, provider_name, top_k)
                if mode == "supplement":
                    curated = _search_curated(query, top_k)
                    return json.dumps(
                        {
                            "items": _merge_supplement_items(curated, items, top_k),
                            "provider": provider_name,
                            "provider_status": _provider_status(raw),
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {"items": items, "provider": provider_name},
                    ensure_ascii=False,
                )
            # Degradation path: a failing OR hanging provider must not block
            # recall — fall back to the built-in curated memory.
            logger.warning(
                "memory provider '%s' search unusable (%s), falling back to curated memory",
                provider_name, failure,
            )
            if mode == "supplement":
                return json.dumps(
                    {
                        "items": _search_curated(query, top_k),
                        "provider": provider_name,
                        "provider_status": _failure_status(failure),
                    },
                    ensure_ascii=False,
                )

        # Layer 2: built-in curated memory (MEMORY.md + USER.md).
        return json.dumps({"items": _search_curated(query, top_k)}, ensure_ascii=False)
    except Exception as exc:
        logger.warning("search_memory failed: %s", exc)
        return json.dumps({"items": []}, ensure_ascii=False)


from tools.registry import registry  # noqa: E402
from tools.memory_tool import check_memory_requirements  # noqa: E402

registry.register(
    name="search_memory",
    toolset="memory",
    schema=SEARCH_MEMORY_SCHEMA,
    handler=search_memory_tool,
    # Same availability gate as the memory tool (always on; the toolset system
    # decides exposure). Registered into the SAME "memory" toolset — via the
    # registry merge, not the static catalog, so platform composite
    # reverse-mapping keeps working (see toolsets issue #49622).
    check_fn=check_memory_requirements,
    # Explicit native memory recall should be directly available. In smart
    # Deep Memory mode this is also the only model-facing recall entry point.
    defer_to_tool_search=False,
    emoji="🔎",
)
