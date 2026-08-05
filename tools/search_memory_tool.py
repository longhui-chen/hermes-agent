"""Search Memory Tool — unified read-only recall over persistent memory.

Two-layer retrieval:
  1. If an activated memory provider object exposes a callable
     ``search(query, top_k)`` method (an OPTIONAL provider extension point —
     none of the in-tree providers implement it today; detected via hasattr,
     skipped when absent), the query is proxied to it.
  2. Otherwise the built-in curated memory is searched: MEMORY.md and USER.md
     entries (``§``-delimited, see tools/memory_tool.py) are scored with a
     case-insensitive token-level match (hit count + full-phrase bonus — no
     external dependencies, no embeddings) and the top_k entries returned.

Result contract (JSON string):
    {"items": [{"id", "source", "excerpt", "score"}, ...], "provider"?: str}

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

logger = logging.getLogger(__name__)

_EXCERPT_MAX_CHARS = 200
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 25

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
                or ""
            )
            try:
                score = float(element.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            items.append({
                "id": str(element.get("id") or _entry_id(text)),
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
    """Return (provider_name, search_callable) for the first active provider
    exposing a callable ``search``, or None. Optional extension point — none
    of the in-tree providers implement it today."""
    if memory_manager is None:
        return None
    try:
        providers = list(getattr(memory_manager, "providers", []) or [])
    except Exception:
        return None
    for provider in providers:
        search_fn = getattr(provider, "search", None)
        if callable(search_fn):
            try:
                name = str(getattr(provider, "name", "") or "")
            except Exception:
                name = ""
            return (name or type(provider).__name__, search_fn)
    return None


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

        # Layer 1: proxy to an activated provider's optional search().
        # ``memory_manager`` is threaded in by the agent runtime
        # (agent/agent_runtime_helpers.py), mirroring how the memory tool
        # receives its store; registry-only dispatch paths fall through to
        # the curated layer.
        found = _find_provider_search(kw.get("memory_manager"))
        if found is not None:
            provider_name, search_fn = found
            try:
                raw = search_fn(query, top_k)
                items = _normalize_provider_items(raw, provider_name, top_k)
                return json.dumps(
                    {"items": items, "provider": provider_name},
                    ensure_ascii=False,
                )
            except Exception as exc:
                # Degradation path: a failing provider must not block recall —
                # fall back to the built-in curated memory.
                logger.warning(
                    "memory provider '%s' search failed, falling back to curated memory: %s",
                    provider_name, exc,
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
    emoji="🔎",
)
