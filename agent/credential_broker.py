"""Client for local-server's process-authenticated AgentComputer credential broker."""

from __future__ import annotations

import json
import os
import re
import socket
import struct
from pathlib import Path


DEFAULT_SOCKET_PATH = Path(
    "/run/zettlab-local-server/agentcomputer-credential.sock"
)
_SOCKET_ENV = "ZETTLAB_AGENTCOMPUTER_CREDENTIAL_SOCKET"
_MAX_REQUEST_BYTES = 1024
_MAX_RESPONSE_BYTES = 8 * 1024
_MAX_AGENT_ID_BYTES = 256
_TOKEN_RE = re.compile(r"[0-9a-f]{64}")
_TIMEOUT_SECONDS = 2.0


def request_agentcomputer_token(
    agent_id: str,
    *,
    socket_path: str | os.PathLike[str] | None = None,
) -> str:
    """Request one short-lived token; the server authenticates this exact PID."""

    normalized_agent_id = str(agent_id or "").strip()
    encoded_agent_id = normalized_agent_id.encode("utf-8")
    if (
        not normalized_agent_id
        or len(encoded_agent_id) > _MAX_AGENT_ID_BYTES
        or normalized_agent_id in {".", ".."}
        or any(
            marker in normalized_agent_id
            for marker in ("/", "\\", "\x00", "\r", "\n")
        )
    ):
        raise RuntimeError("AgentComputer broker profile identity is invalid")

    selected_path = Path(
        socket_path
        or os.environ.get(_SOCKET_ENV, "")
        or DEFAULT_SOCKET_PATH
    )
    if not selected_path.is_absolute():
        raise RuntimeError("AgentComputer broker socket path is invalid")
    payload = json.dumps(
        {"agent_id": normalized_agent_id, "purpose": "agentcomputer"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > _MAX_REQUEST_BYTES:
        raise RuntimeError("AgentComputer broker request is invalid")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(_TIMEOUT_SECONDS)
            client.connect(str(selected_path))
            client.sendall(struct.pack(">I", len(payload)) + payload)
            response_size = struct.unpack(">I", _recv_exact(client, 4))[0]
            if response_size == 0 or response_size > _MAX_RESPONSE_BYTES:
                raise RuntimeError("AgentComputer broker response is invalid")
            response_payload = _recv_exact(client, response_size)
    except (OSError, TimeoutError, struct.error) as exc:
        raise RuntimeError("AgentComputer credential broker is unavailable") from exc

    try:
        response = json.loads(response_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("AgentComputer broker response is invalid") from exc
    if not isinstance(response, dict) or set(response) - {"token", "error"}:
        raise RuntimeError("AgentComputer broker response is invalid")
    token = response.get("token")
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise RuntimeError("AgentComputer credential broker denied the request")
    return token


def _recv_exact(client: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = client.recv(remaining)
        if not chunk:
            raise RuntimeError("AgentComputer broker closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
