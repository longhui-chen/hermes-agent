import json

from gateway.session_seen_models import load_seen_models, save_seen_models


def test_roundtrip(tmp_path):
    p = tmp_path / "seen.json"
    save_seen_models({"s1": "glm-5", "s2": "deepseek-v4"}, path=p)
    assert load_seen_models(p) == {"s1": "glm-5", "s2": "deepseek-v4"}


def test_load_missing_returns_empty(tmp_path):
    assert load_seen_models(tmp_path / "nope.json") == {}


def test_load_malformed_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert load_seen_models(p) == {}


def test_load_filters_non_string_entries(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text(
        json.dumps({"ok": "m", "bad_val": 123, "": "empty-key", "bad_key": ""}),
        encoding="utf-8",
    )
    assert load_seen_models(p) == {"ok": "m"}


def test_save_is_atomic_no_tmp_left(tmp_path):
    p = tmp_path / "seen.json"
    save_seen_models({"s1": "m"}, path=p)
    assert p.exists()
    assert not (tmp_path / "seen.json.tmp").exists()
