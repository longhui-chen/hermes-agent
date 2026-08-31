import json
import socket
import struct
import tempfile
import threading
from pathlib import Path

import pytest

from agent.credential_broker import (
    request_agentcomputer_token,
    request_app_auto_refresh_token,
    request_lark_cli,
)


def _serve_once(path, response, *, purpose="agentcomputer", expected=None):
    ready = threading.Event()

    def run():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                size = struct.unpack(">I", connection.recv(4))[0]
                received = bytearray()
                while len(received) < size:
                    chunk = connection.recv(size - len(received))
                    if not chunk:
                        break
                    received.extend(chunk)
                request = json.loads(received)
                expected_request = {
                    "agent_id": "agent-1",
                    "purpose": purpose,
                }
                if expected:
                    expected_request.update(expected)
                assert request == expected_request
                payload = json.dumps(response, separators=(",", ":")).encode()
                connection.sendall(struct.pack(">I", len(payload)) + payload)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread


def test_request_agentcomputer_token_uses_bounded_unix_protocol():
    with tempfile.TemporaryDirectory(prefix="acb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        token = "a" * 64
        thread = _serve_once(path, {"token": token})
        assert request_agentcomputer_token("agent-1", socket_path=path) == token
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_request_app_auto_refresh_token_uses_dedicated_purpose():
    with tempfile.TemporaryDirectory(prefix="acb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        token = "b" * 64
        binding = {
            "operation_kind": "apphost_publish_v1",
            "operation": {"operation_id": "op-1", "data_refresh": "user_confirmed_auto"},
            "owner_principal": "iam:user-1",
            "owner_agent_id": "agent-1",
            "turn_id": "turn-1",
            "session_id": "session-1",
        }
        thread = _serve_once(path, {"token": token}, purpose="app-auto-refresh", expected=binding)
        assert request_app_auto_refresh_token("agent-1", socket_path=path, **binding) == token
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_request_app_auto_refresh_token_preserves_one_megabyte_business_limit():
    with tempfile.TemporaryDirectory(prefix="acb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        token = "c" * 64
        binding = {
            "operation_kind": "app_dedicated_create_v1",
            "operation": {"soul_identity": "x" * (300 * 1024)},
            "owner_principal": "iam:user-1",
            "owner_agent_id": "agent-1",
            "turn_id": "turn-1",
            "session_id": "session-1",
        }
        thread = _serve_once(
            path,
            {"token": token},
            purpose="app-auto-refresh",
            expected=binding,
        )
        assert (
            request_app_auto_refresh_token("agent-1", socket_path=path, **binding)
            == token
        )
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_request_agentcomputer_token_rejects_denial_and_bad_profile():
    with tempfile.TemporaryDirectory(prefix="acb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        thread = _serve_once(path, {"error": "denied"})
        with pytest.raises(RuntimeError, match="denied"):
            request_agentcomputer_token("agent-1", socket_path=path)
        thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="identity"):
        request_agentcomputer_token("../agent", socket_path=path)


def test_request_lark_cli_keeps_credentials_server_side():
    with tempfile.TemporaryDirectory(prefix="lcb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        ready = threading.Event()

        def run():
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                server.listen(1)
                ready.set()
                connection, _ = server.accept()
                with connection:
                    size = struct.unpack(">I", connection.recv(4))[0]
                    request = json.loads(connection.recv(size))
                    assert request == {
                        "agent_id": "agent-1",
                        "purpose": "lark-cli",
                        "args": [
                            "mail",
                            "user_mailbox.messages",
                            "list",
                            "--user-mailbox-id",
                            "me",
                        ],
                        "timeout_seconds": 45,
                    }
                    payload = json.dumps(
                        {
                            "output": '{"ok":true}',
                            "exit_code": 0,
                            "timed_out": False,
                        },
                        separators=(",", ":"),
                    ).encode()
                    connection.sendall(
                        struct.pack(">I", len(payload)) + payload
                    )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert ready.wait(2)
        result = request_lark_cli(
            "agent-1",
            [
                "mail",
                "user_mailbox.messages",
                "list",
                "--user-mailbox-id",
                "me",
            ],
            timeout_seconds=45,
            socket_path=path,
        )
        assert result.output == '{"ok":true}'
        assert result.exit_code == 0
        assert result.timed_out is False
        thread.join(timeout=2)
        assert not thread.is_alive()
