from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from agent import secret_scope
from gateway.session_context import clear_session_vars, set_session_vars
from tools import tts_tool


async def _write_edge_output(_text: str, output_path: str, _tts_config: dict) -> str:
    Path(output_path).write_bytes(b"mp3")
    return output_path


@contextmanager
def _managed_zettlab_session(output_root: Path, session_id: str):
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope(
        {"ZET_AGENT_OUTPUT_DIR": str(output_root)}
    )
    session_tokens = set_session_vars(
        platform="zet_agent",
        session_id=session_id,
        async_delivery=False,
    )
    try:
        yield
    finally:
        clear_session_vars(session_tokens)
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


def _stub_edge_tts(monkeypatch) -> None:
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_get_provider", lambda _config: "edge")
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_edge_output)


def test_managed_zettlab_tts_defaults_to_session_output_dir(tmp_path, monkeypatch):
    output_root = tmp_path / "agent-output"
    output_root.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    _stub_edge_tts(monkeypatch)

    with _managed_zettlab_session(
        output_root,
        "zettlab:local-dev:main:session-abc",
    ):
        result = json.loads(tts_tool.text_to_speech_tool("hello"))

    file_path = Path(result["file_path"])
    assert result["success"] is True
    assert file_path.parent == output_root / "session-abc"
    assert file_path.read_bytes() == b"mp3"
    assert result["media_tag"] == f"MEDIA:{file_path}"


def test_managed_zettlab_tts_resolves_output_root_per_profile(tmp_path, monkeypatch):
    output_a = tmp_path / "agent-a-output"
    output_b = tmp_path / "agent-b-output"
    output_a.mkdir()
    output_b.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    _stub_edge_tts(monkeypatch)

    with _managed_zettlab_session(output_a, "zettlab:local-dev:a:session-a"):
        result_a = json.loads(tts_tool.text_to_speech_tool("profile a"))
    with _managed_zettlab_session(output_b, "zettlab:local-dev:b:session-b"):
        result_b = json.loads(tts_tool.text_to_speech_tool("profile b"))

    assert Path(result_a["file_path"]).parent == output_a / "session-a"
    assert Path(result_b["file_path"]).parent == output_b / "session-b"


def test_managed_zettlab_tts_rejects_unsafe_session_bucket(tmp_path, monkeypatch):
    output_root = tmp_path / "agent-output"
    output_root.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    _stub_edge_tts(monkeypatch)

    with _managed_zettlab_session(
        output_root,
        "zettlab:local-dev:main:../../escape",
    ):
        result = json.loads(tts_tool.text_to_speech_tool("hello"))

    file_path = Path(result["file_path"])
    assert result["success"] is True
    assert file_path.parent == output_root
    assert file_path.is_relative_to(output_root)


def test_managed_zettlab_tts_rejects_dot_session_buckets(tmp_path, monkeypatch):
    output_root = tmp_path / "agent-output"
    output_root.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    _stub_edge_tts(monkeypatch)

    for bucket in (".", ".."):
        with _managed_zettlab_session(
            output_root,
            f"zettlab:local-dev:main:{bucket}",
        ):
            result = json.loads(tts_tool.text_to_speech_tool("hello"))

        file_path = Path(result["file_path"])
        assert result["success"] is True
        assert file_path.parent == output_root.resolve()
        assert file_path.is_relative_to(output_root.resolve())


def test_non_zettlab_tts_keeps_existing_cache_default(tmp_path, monkeypatch):
    output_root = tmp_path / "agent-output"
    cache_root = tmp_path / "cache" / "audio"
    output_root.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(cache_root))
    _stub_edge_tts(monkeypatch)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope(
        {"ZET_AGENT_OUTPUT_DIR": str(output_root)}
    )
    session_tokens = set_session_vars(
        platform="discord",
        session_id="zettlab:local-dev:main:session-abc",
    )
    try:
        result = json.loads(tts_tool.text_to_speech_tool("hello"))
    finally:
        clear_session_vars(session_tokens)
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert Path(result["file_path"]).parent == cache_root
