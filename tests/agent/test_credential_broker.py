import json
import socket
import struct
import tempfile
import threading
from pathlib import Path

import pytest

from agent.credential_broker import request_agentcomputer_token


def _serve_once(path, response):
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
                    "purpose": "agentcomputer",
                }
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


def test_request_agentcomputer_token_rejects_denial_and_bad_profile():
    with tempfile.TemporaryDirectory(prefix="acb-", dir="/tmp") as directory:
        path = Path(directory) / "credential.sock"
        thread = _serve_once(path, {"error": "denied"})
        with pytest.raises(RuntimeError, match="denied"):
            request_agentcomputer_token("agent-1", socket_path=path)
        thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="identity"):
        request_agentcomputer_token("../agent", socket_path=path)
