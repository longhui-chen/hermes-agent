"""Tests for gateway/shutdown_flush.py — pending message durability (#72680)."""

import json
import os
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.shutdown_flush import (
    _serialise_value,
    flush_pending_to_file,
    recover_pending_to_db,
)
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_flush_dir(tmp_path: Path) -> Path:
    """Create a temp flush dir and monkeypatch _get_flush_dir to use it."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    return flush_dir


def test_flush_writes_string_pending_to_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    pending = {"agent:main:telegram:supergroup:123": "hello world"}
    count = flush_pending_to_file(pending, reason="shutdown")
    assert count == 1
    files = list(flush_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["session_key"] == "agent:main:telegram:supergroup:123"
    assert payload["reason"] == "shutdown"
    assert payload["data"]["text"] == "hello world"
    assert ":" not in files[0].name
    assert "telegram" not in files[0].name


def test_flush_writes_message_event_to_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    event = MagicMock()
    event.text = "user message"
    event.session_id = "20260728_120000_abc"
    event.platform = "telegram"
    event.sender_id = "456"
    event.sender_name = "Alice"
    event.reply_to = None
    event.media = None
    event.raw_event = None

    count = flush_pending_to_file({"session_key_1": event}, reason="adapter_shutdown")
    assert count == 1
    files = list(flush_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["data"]["text"] == "user message"
    assert payload["data"]["session_id"] == "20260728_120000_abc"


def test_recover_inserts_via_append_message_and_deletes_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    ts = int(time.time())
    # Write a flush file with session_id
    payload = {
        "session_key": "agent:main:telegram:supergroup:123",
        "reason": "shutdown",
        "ts": ts,
        "data": {
            "text": "lost message",
            "session_id": "20260728_120000_abc",
        },
    }
    flush_file = flush_dir / "test_session_123.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(mock_db)

    assert count == 1
    mock_db.append_message.assert_called_once_with(
        session_id="20260728_120000_abc",
        role="user",
        content="lost message",
        timestamp=ts,
    )
    assert not flush_file.exists()


def test_serialise_object_with_text():
    obj = MagicMock()
    obj.text = "msg"
    obj.session_id = "sid"
    obj.platform = None
    obj.sender_id = None
    obj.sender_name = None
    obj.reply_to = None
    obj.media = None
    obj.raw_event = None
    result = _serialise_value(obj)
    assert result is not None
    assert result["text"] == "msg"
    assert result["session_id"] == "sid"


def test_cross_sender_fifo_flush_and_recovery_preserve_every_event(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_id = "20260819_065133_fifo"

    def event(index: int, kind: MessageType, media_url: str = "") -> MessageEvent:
        item = MessageEvent(
            text=f"message-{index}",
            message_type=kind,
            source=SessionSource(
                platform=Platform("teams"),
                chat_id="shared-room",
                chat_type="group",
                user_id=f"sender-{index}",
            ),
            media_urls=[media_url] if media_url else [],
            media_types=["image/png"] if media_url else [],
            message_id=f"platform-{index}",
        )
        return item

    first = event(1, MessageType.TEXT)
    second = event(2, MessageType.PHOTO, "/cache/two.png")
    third = event(3, MessageType.DOCUMENT, "/cache/three.pdf")
    first._gateway_pending_event_queue = [second, third]

    session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={"shared-session": SimpleNamespace(session_id=session_id)},
        _ensure_loaded_locked=lambda: None,
    )
    assert flush_pending_to_file(
        {"shared-session": first}, session_store=session_store,
    ) == 1
    payload_path = next(flush_dir.glob("*.json"))
    data = json.loads(payload_path.read_text())["data"]
    assert [item["text"] for item in data["events"]] == [
        "message-1", "message-2", "message-3"
    ]
    assert data["events"][1]["source"]["user_id"] == "sender-2"
    assert data["events"][1]["media_urls"] == ["/cache/two.png"]
    assert data["events"][1]["media_types"] == ["image/png"]

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 3
    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "message-1", "message-2", "message-3"
    ]
    restored_second = mock_db.append_message.call_args_list[1].kwargs
    assert restored_second["display_metadata"]["source"]["user_id"] == "sender-2"
    assert restored_second["display_metadata"]["media_types"] == ["image/png"]
    assert restored_second["display_metadata"]["media_names"] == ["two.png"]
    assert "media_urls" not in restored_second["display_metadata"]
    assert restored_second["api_content"].endswith("[file:/cache/two.png]")
    assert not payload_path.exists()


def test_get_flush_dir_uses_get_hermes_home(tmp_path, monkeypatch):
    """Flush dir must use get_hermes_home(), not hardcoded Path.home()."""
    import gateway.shutdown_flush as mod

    captured = {}

    def fake_get_hermes_home():
        from pathlib import Path
        captured["called"] = True
        return tmp_path

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", fake_get_hermes_home
    )
    result = mod._get_flush_dir()
    assert captured.get("called") is True
    assert result == tmp_path / "pending_messages"
