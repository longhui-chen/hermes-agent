"""Tests for tools/search_memory_tool.py."""

import hashlib
import json

import pytest

from tools.search_memory_tool import search_memory_tool


ENTRY_DELIMITER = "\n§\n"


def _write_memory_files(tmp_path, monkeypatch, memory_entries=(), user_entries=()):
    """Point the curated-memory dir at tmp_path and write §-delimited files."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    if memory_entries:
        (tmp_path / "MEMORY.md").write_text(
            ENTRY_DELIMITER.join(memory_entries), encoding="utf-8"
        )
    if user_entries:
        (tmp_path / "USER.md").write_text(
            ENTRY_DELIMITER.join(user_entries), encoding="utf-8"
        )


def _call(query, top_k=None, **kw):
    args = {"query": query}
    if top_k is not None:
        args["top_k"] = top_k
    out = search_memory_tool(args, **kw)
    # Tool returns a JSON STRING (same contract as memory/list_my_channels).
    assert isinstance(out, str)
    return json.loads(out)


# --- Curated path: scoring, ranking, truncation -----------------------------


def test_hit_ranking_more_token_hits_rank_higher(tmp_path, monkeypatch):
    _write_memory_files(
        tmp_path, monkeypatch,
        memory_entries=[
            "Metro dev-server logs live in /tmp; restart with the start skill",
            "Metro dev-server for zettlab-app: logs, port 8081, dev-client",
            "Unrelated note about calendars",
        ],
    )
    parsed = _call("metro logs dev-client")
    items = parsed["items"]
    assert len(items) == 2  # the calendar note must not match
    # Entry with 3 token hits outranks the one with 2.
    assert "port 8081" in items[0]["excerpt"]
    assert items[0]["score"] > items[1]["score"]


def test_full_phrase_match_outranks_scattered_tokens(tmp_path, monkeypatch):
    _write_memory_files(
        tmp_path, monkeypatch,
        memory_entries=[
            # Both tokens present but scattered:
            "the hot path is an update of the SSE contract",
            # Verbatim phrase:
            "hot update goes through Pushy, no new native shell",
        ],
    )
    items = _call("hot update")["items"]
    assert len(items) == 2
    assert "Pushy" in items[0]["excerpt"]
    assert items[0]["score"] > items[1]["score"]


def test_matching_is_case_insensitive(tmp_path, monkeypatch):
    _write_memory_files(
        tmp_path, monkeypatch,
        memory_entries=["JPUSH_APP_KEY must be present in the build env"],
    )
    items = _call("jpush_app_key")["items"]
    assert len(items) == 1


def test_top_k_truncation_and_default(tmp_path, monkeypatch):
    _write_memory_files(
        tmp_path, monkeypatch,
        memory_entries=[f"zettlab convention number {i}" for i in range(10)],
    )
    assert len(_call("zettlab convention")["items"]) == 5  # default top_k
    assert len(_call("zettlab convention", top_k=3)["items"]) == 3
    assert len(_call("zettlab convention", top_k=0)["items"]) == 1  # clamped to >= 1
    assert len(_call("zettlab convention", top_k="bogus")["items"]) == 5  # coerce fallback


def test_user_md_entries_carry_user_source(tmp_path, monkeypatch):
    _write_memory_files(
        tmp_path, monkeypatch,
        memory_entries=["zettlab build convention"],
        user_entries=["prefers concise Chinese replies for zettlab work"],
    )
    items = _call("zettlab")["items"]
    assert {item["source"] for item in items} == {"memory", "user"}


def test_excerpt_truncated_to_200_chars_and_id_is_content_sha1(tmp_path, monkeypatch):
    long_entry = "zettlab " + "x" * 400
    _write_memory_files(tmp_path, monkeypatch, memory_entries=[long_entry])
    items = _call("zettlab")["items"]
    assert len(items) == 1
    assert len(items[0]["excerpt"]) == 201  # 200 chars + ellipsis
    assert items[0]["excerpt"].endswith("…")
    # id = sha1(full entry)[:12] — stable content hash for future citations.
    assert items[0]["id"] == hashlib.sha1(long_entry.encode("utf-8")).hexdigest()[:12]


# --- Curated path: degradation ----------------------------------------------


def test_empty_query_returns_no_items(tmp_path, monkeypatch):
    _write_memory_files(tmp_path, monkeypatch, memory_entries=["anything"])
    assert _call("")["items"] == []
    assert _call("   ")["items"] == []


def test_missing_memory_files_return_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    parsed = _call("anything at all")
    assert parsed == {"items": []}


def test_unreadable_memory_dir_never_raises(monkeypatch):
    def _boom():
        raise OSError("no such profile dir")

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", _boom)
    parsed = _call("anything")
    assert parsed == {"items": []}


def test_malformed_args_never_raise():
    out = search_memory_tool(None)
    assert json.loads(out) == {"items": []}
    out = search_memory_tool({"query": 123, "top_k": "NaN"})
    assert "items" in json.loads(out)


# --- Provider proxy path -----------------------------------------------------


class _FakeSearchProvider:
    """Provider exposing the OPTIONAL search(query, top_k) extension point."""

    name = "fake-recall"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return self._result


class _FakeManager:
    def __init__(self, *providers):
        self.providers = list(providers)


class _NoSearchProvider:
    name = "no-search"


def test_provider_with_search_is_proxied():
    provider = _FakeSearchProvider(result=[
        {"id": "abc123", "source": "graph", "excerpt": "user prefers dark mode", "score": 0.9},
        {"text": "works on zettlab-app", "score": "0.5"},
    ])
    parsed = _call("preferences", top_k=2, memory_manager=_FakeManager(provider))
    assert parsed["provider"] == "fake-recall"
    assert provider.calls == [("preferences", 2)]
    assert parsed["items"][0] == {
        "id": "abc123", "source": "graph",
        "excerpt": "user prefers dark mode", "score": 0.9,
    }
    # Missing fields are normalized: id from content hash, source from provider,
    # non-numeric-safe score coercion.
    assert parsed["items"][1]["source"] == "fake-recall"
    assert parsed["items"][1]["excerpt"] == "works on zettlab-app"
    assert parsed["items"][1]["score"] == 0.5


def test_provider_without_search_falls_through_to_curated(tmp_path, monkeypatch):
    _write_memory_files(tmp_path, monkeypatch, memory_entries=["zettlab curated fact"])
    parsed = _call("zettlab", memory_manager=_FakeManager(_NoSearchProvider()))
    # hasattr probe finds nothing callable -> curated layer answers.
    assert "provider" not in parsed
    assert parsed["items"][0]["excerpt"] == "zettlab curated fact"
    assert parsed["items"][0]["source"] == "memory"


def test_provider_search_exception_degrades_to_curated(tmp_path, monkeypatch):
    _write_memory_files(tmp_path, monkeypatch, memory_entries=["zettlab curated fact"])
    provider = _FakeSearchProvider(error=RuntimeError("backend down"))
    parsed = _call("zettlab", memory_manager=_FakeManager(provider))
    # HARD CONSTRAINT: recall failure never blocks — no exception escapes,
    # and the curated layer still answers.
    assert parsed["items"][0]["excerpt"] == "zettlab curated fact"
    assert "provider" not in parsed


def test_provider_garbage_result_yields_empty_items():
    provider = _FakeSearchProvider(result="not-a-list")
    parsed = _call("anything", memory_manager=_FakeManager(provider))
    assert parsed == {"items": [], "provider": "fake-recall"}


def test_provider_results_respect_top_k():
    provider = _FakeSearchProvider(result=[{"text": f"item {i}"} for i in range(10)])
    parsed = _call("items", top_k=4, memory_manager=_FakeManager(provider))
    assert len(parsed["items"]) == 4


# --- Wiring ------------------------------------------------------------------


def test_registered_in_memory_toolset_with_memory_gate():
    """Live-path wiring: the tool is registered, rides in the SAME toolset as
    the memory tool (registry merge), and shares its availability gate."""
    from tools.memory_tool import check_memory_requirements
    from tools.registry import registry
    import tools.search_memory_tool  # noqa: F401 — self-registers on import
    import toolsets

    entry = registry.get_entry("search_memory")
    assert entry is not None
    assert entry.toolset == "memory"
    assert entry.check_fn is check_memory_requirements
    assert "search_memory" in toolsets.resolve_toolset("memory")
    # The static catalog view must stay untouched so platform composite
    # reverse-mapping (issue #49622) keeps inferring the memory toolset.
    assert "search_memory" not in toolsets.resolve_toolset(
        "memory", include_registry=False
    )
