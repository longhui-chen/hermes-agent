"""Client for local-server's process-authenticated AgentComputer credential broker."""

from __future__ import annotations

import json
import os
import re
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SOCKET_PATH = Path(
    "/run/zettlab-local-server/agentcomputer-credential.sock"
)
_SOCKET_ENV = "ZETTLAB_AGENTCOMPUTER_CREDENTIAL_SOCKET"
_MAX_REQUEST_BYTES = 256 * 1024
_MAX_TOKEN_RESPONSE_BYTES = 8 * 1024
_MAX_LARK_RESPONSE_BYTES = 512 * 1024
_MAX_AGENT_ID_BYTES = 256
_MAX_LARK_ARGUMENTS = 128
_MAX_LARK_ARGUMENT_BYTES = 16 * 1024
_MAX_LARK_TIMEOUT_SECONDS = 600
_TOKEN_RE = re.compile(r"[0-9a-f]{64}")
_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class LarkCLIResult:
    output: str
    exit_code: int
    timed_out: bool = False


def request_agentcomputer_token(
    agent_id: str,
    *,
    socket_path: str | os.PathLike[str] | None = None,
) -> str:
    """Request one short-lived token; the server authenticates this exact PID."""

    return _request_scoped_token(
        agent_id, "agentcomputer", socket_path=socket_path
    )


def request_app_auto_refresh_token(
    agent_id: str,
    *,
    operation_digest: str,
    owner_agent_id: str,
    turn_id: str,
    session_id: str,
    socket_path: str | os.PathLike[str] | None = None,
) -> str:
    """Request a scope bound to one server-attested operation and turn."""
    fields = (operation_digest, owner_agent_id, turn_id, session_id)
    if any(not isinstance(value, str) or not value.strip() for value in fields):
        raise RuntimeError("App Host operation binding is incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", operation_digest):
        raise RuntimeError("App Host operation digest is invalid")

    return _request_scoped_token(
        agent_id,
        "app-auto-refresh",
        operation_digest=operation_digest,
        owner_agent_id=owner_agent_id,
        turn_id=turn_id,
        session_id=session_id,
        socket_path=socket_path,
    )


def _request_scoped_token(
    agent_id: str,
    purpose: str,
    *,
    socket_path: str | os.PathLike[str] | None = None,
    **binding: str,
) -> str:

    normalized_agent_id = _normalize_agent_id(agent_id)
    payload = json.dumps(
        {"agent_id": normalized_agent_id, "purpose": purpose, **binding},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    response = _exchange(
        payload,
        socket_path=socket_path,
        response_limit=_MAX_TOKEN_RESPONSE_BYTES,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if not isinstance(response, dict) or set(response) - {"token", "error"}:
        raise RuntimeError("AgentComputer broker response is invalid")
    token = response.get("token")
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise RuntimeError("AgentComputer credential broker denied the request")
    return token


def request_lark_cli(
    agent_id: str,
    args: Sequence[str],
    *,
    timeout_seconds: int,
    socket_path: str | os.PathLike[str] | None = None,
) -> LarkCLIResult:
    """Run one lark-cli argv through the process-authenticated local broker.

    The Hermes gateway submits only arguments. The local server chooses the
    executable, profile HOME, and working directory, so OAuth material never
    enters a model-authored shell environment or agent-writable directory.
    """

    normalized_agent_id = _normalize_agent_id(agent_id)
    try:
        normalized_timeout = int(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lark-cli broker timeout is invalid") from exc
    if not 1 <= normalized_timeout <= _MAX_LARK_TIMEOUT_SECONDS:
        raise RuntimeError("lark-cli broker timeout is invalid")
    if isinstance(args, (str, bytes)):
        raise RuntimeError("lark-cli broker arguments are invalid")
    normalized_args = [str(value) for value in args]
    if len(normalized_args) > _MAX_LARK_ARGUMENTS or any(
        "\x00" in value
        or len(value.encode("utf-8")) > _MAX_LARK_ARGUMENT_BYTES
        for value in normalized_args
    ):
        raise RuntimeError("lark-cli broker arguments are invalid")

    payload = json.dumps(
        {
            "agent_id": normalized_agent_id,
            "purpose": "lark-cli",
            "args": normalized_args,
            "timeout_seconds": normalized_timeout,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    response = _exchange(
        payload,
        socket_path=socket_path,
        response_limit=_MAX_LARK_RESPONSE_BYTES,
        timeout_seconds=normalized_timeout + _TIMEOUT_SECONDS,
    )
    if not isinstance(response, dict) or set(response) - {
        "output",
        "exit_code",
        "timed_out",
        "error",
    }:
        raise RuntimeError("lark-cli broker response is invalid")
    error = response.get("error", "")
    if error:
        if not isinstance(error, str):
            raise RuntimeError("lark-cli broker response is invalid")
        raise RuntimeError(error)
    output = response.get("output", "")
    exit_code = response.get("exit_code", 0)
    timed_out = response.get("timed_out", False)
    if (
        not isinstance(output, str)
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not -1 <= exit_code <= 255
        or not isinstance(timed_out, bool)
    ):
        raise RuntimeError("lark-cli broker response is invalid")
    return LarkCLIResult(
        output=output,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def request_lark_auth_start(
    agent_id: str,
    *,
    scope: str = "",
    timeout_seconds: int,
    socket_path: str | os.PathLike[str] | None = None,
) -> LarkCLIResult:
    """Start the device-code OAuth flow with server-built argv.

    The agent supplies only a scope list; zls composes the exact
    `auth login --no-wait --json [--scope ...]` invocation, so authorization
    can never be blocked by (or smuggled through) argv matching.
    """

    return _request_lark_auth(
        agent_id,
        {"purpose": "lark-auth-start", "scope": str(scope or "")},
        timeout_seconds=timeout_seconds,
        socket_path=socket_path,
    )


def request_lark_auth_complete(
    agent_id: str,
    *,
    device_code: str,
    timeout_seconds: int,
    socket_path: str | os.PathLike[str] | None = None,
) -> LarkCLIResult:
    """Finish the device-code OAuth flow; tokens land in the service home."""

    return _request_lark_auth(
        agent_id,
        {"purpose": "lark-auth-complete", "device_code": str(device_code or "")},
        timeout_seconds=timeout_seconds,
        socket_path=socket_path,
    )


def _request_lark_auth(
    agent_id: str,
    body: dict,
    *,
    timeout_seconds: int,
    socket_path: str | os.PathLike[str] | None,
) -> LarkCLIResult:
    normalized_agent_id = _normalize_agent_id(agent_id)
    try:
        normalized_timeout = int(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lark-cli broker timeout is invalid") from exc
    if not 1 <= normalized_timeout <= _MAX_LARK_TIMEOUT_SECONDS:
        raise RuntimeError("lark-cli broker timeout is invalid")
    payload = json.dumps(
        {
            "agent_id": normalized_agent_id,
            "timeout_seconds": normalized_timeout,
            **body,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    response = _exchange(
        payload,
        socket_path=socket_path,
        response_limit=_MAX_LARK_RESPONSE_BYTES,
        timeout_seconds=normalized_timeout + _TIMEOUT_SECONDS,
    )
    return _parse_lark_result(response)


def _parse_lark_result(response: object) -> LarkCLIResult:
    if not isinstance(response, dict) or set(response) - {
        "output",
        "exit_code",
        "timed_out",
        "error",
    }:
        raise RuntimeError("lark-cli broker response is invalid")
    error = response.get("error", "")
    if error:
        if not isinstance(error, str):
            raise RuntimeError("lark-cli broker response is invalid")
        raise RuntimeError(error)
    output = response.get("output", "")
    exit_code = response.get("exit_code", 0)
    timed_out = response.get("timed_out", False)
    if (
        not isinstance(output, str)
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not -1 <= exit_code <= 255
        or not isinstance(timed_out, bool)
    ):
        raise RuntimeError("lark-cli broker response is invalid")
    return LarkCLIResult(output=output, exit_code=exit_code, timed_out=timed_out)


def _normalize_agent_id(agent_id: str) -> str:
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
    return normalized_agent_id


def _exchange(
    payload: bytes,
    *,
    socket_path: str | os.PathLike[str] | None,
    response_limit: int,
    timeout_seconds: float,
) -> object:
    if not payload or len(payload) > _MAX_REQUEST_BYTES:
        raise RuntimeError("AgentComputer broker request is invalid")
    selected_path = Path(
        socket_path
        or os.environ.get(_SOCKET_ENV, "")
        or DEFAULT_SOCKET_PATH
    )
    if not selected_path.is_absolute():
        raise RuntimeError("AgentComputer broker socket path is invalid")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(str(selected_path))
            client.sendall(struct.pack(">I", len(payload)) + payload)
            response_size = struct.unpack(">I", _recv_exact(client, 4))[0]
            if response_size == 0 or response_size > response_limit:
                raise RuntimeError("AgentComputer broker response is invalid")
            response_payload = _recv_exact(client, response_size)
    except (OSError, TimeoutError, struct.error) as exc:
        raise RuntimeError("AgentComputer credential broker is unavailable") from exc

    try:
        return json.loads(response_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("AgentComputer broker response is invalid") from exc


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
