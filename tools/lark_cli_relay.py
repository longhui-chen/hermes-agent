"""Per-agent relay that lets sandboxed subprocesses reach the lark-cli broker.

Skill-pack scripts invoke ``lark-cli`` from *inside* python subprocesses, so the
terminal tool's command-string interception never sees them.  This relay closes
that gap without handing any credential to the sandbox:

- the gateway (a registered broker peer) listens on one Unix socket per agent
  UID under ``/run/zettlab-claw/lark-relay/``;
- a root-owned ``lark-cli`` shim is prepended to the sandbox PATH; it forwards
  argv over the socket and prints the brokered result;
- every accepted connection is authenticated with ``SO_PEERCRED`` — only the
  exact sandbox UID the relay was started for may talk to it;
- ``auth login`` device-code forms are translated to the dedicated broker
  purposes (server-built argv); everything else rides the ``lark-cli`` purpose
  and the broker's management-deny model.

The relay never parses credential material; it moves argv one way and
already-redacted CLI output the other way.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

RELAY_ROOT = Path("/run/zettlab-claw/lark-relay")
SOCKET_ENV = "LARK_CLI_BROKER_SOCKET"
_MAX_FRAME_BYTES = 256 * 1024
_MAX_OUTPUT_BYTES = 512 * 1024
_DEFAULT_TIMEOUT_SECONDS = 240
_MAX_TIMEOUT_SECONDS = 600
_IO_TIMEOUT_SECONDS = 5.0
_MAX_CLIENTS = 4

_SHIM_SOURCE = """#!/usr/bin/env python3
# lark-cli sandbox shim: forwards argv to the gateway relay; no credentials here.
import json, os, socket, struct, sys

def main():
    path = os.environ.get("LARK_CLI_BROKER_SOCKET", "")
    if not path:
        sys.stderr.write("lark-cli relay socket is not configured\\n")
        return 1
    payload = json.dumps({"args": sys.argv[1:]}, ensure_ascii=False).encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(620)
            client.connect(path)
            client.sendall(struct.pack(">I", len(payload)) + payload)
            head = b""
            while len(head) < 4:
                chunk = client.recv(4 - len(head))
                if not chunk:
                    raise OSError("relay closed")
                head += chunk
            size = struct.unpack(">I", head)[0]
            body = b""
            while len(body) < size:
                chunk = client.recv(min(65536, size - len(body)))
                if not chunk:
                    raise OSError("relay closed")
                body += chunk
    except OSError as exc:
        sys.stderr.write("lark-cli relay unavailable: %s\\n" % exc)
        return 1
    try:
        reply = json.loads(body)
    except ValueError:
        sys.stderr.write("lark-cli relay reply is invalid\\n")
        return 1
    error = reply.get("error")
    if error:
        sys.stderr.write(str(error) + "\\n")
        return 1
    output = reply.get("output") or ""
    if output:
        sys.stdout.write(output)
        if not output.endswith("\\n"):
            sys.stdout.write("\\n")
    if reply.get("timed_out"):
        sys.stderr.write("Command timed out while running lark-cli\\n")
        return 124
    code = reply.get("exit_code", 0)
    return code if isinstance(code, int) and 0 <= code <= 255 else 1

if __name__ == "__main__":
    raise SystemExit(main())
"""

_relays: dict[int, "_Relay"] = {}
_relays_lock = threading.Lock()


class _Relay:
    def __init__(self, agent_id: str, uid: int, socket_path: Path):
        self.agent_id = agent_id
        self.uid = uid
        self.socket_path = socket_path
        self._listener: socket.socket | None = None
        self._slots = threading.BoundedSemaphore(_MAX_CLIENTS)

    def start(self) -> None:
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        # Connect requires write on the socket inode; the dir chain is 711 and
        # SO_PEERCRED pins the caller, so a permissive inode mode is safe.
        os.chmod(self.socket_path, 0o666)
        listener.listen(_MAX_CLIENTS)
        self._listener = listener
        thread = threading.Thread(
            target=self._serve, name=f"lark-relay-{self.uid}", daemon=True
        )
        thread.start()

    def alive(self) -> bool:
        return self._listener is not None and self.socket_path.exists()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(
                target=self._handle_and_release, args=(conn,), daemon=True
            ).start()

    def _handle_and_release(self, conn: socket.socket) -> None:
        try:
            self._handle(conn)
        except Exception:
            logger.warning("lark relay request failed", exc_info=True)
        finally:
            conn.close()
            self._slots.release()

    def _handle(self, conn: socket.socket) -> None:
        creds = conn.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        _, peer_uid, _ = struct.unpack("3i", creds)
        if peer_uid != self.uid:
            _reply(conn, {"error": "lark-cli relay peer is not authorized"})
            return
        conn.settimeout(_IO_TIMEOUT_SECONDS)
        head = _recv_exact(conn, 4)
        size = struct.unpack(">I", head)[0]
        if size == 0 or size > _MAX_FRAME_BYTES:
            _reply(conn, {"error": "lark-cli relay request is invalid"})
            return
        try:
            request = json.loads(_recv_exact(conn, size))
        except ValueError:
            _reply(conn, {"error": "lark-cli relay request is invalid"})
            return
        args = request.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            _reply(conn, {"error": "lark-cli relay request is invalid"})
            return
        timeout = request.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        if not isinstance(timeout, int) or not 1 <= timeout <= _MAX_TIMEOUT_SECONDS:
            timeout = _DEFAULT_TIMEOUT_SECONDS
        conn.settimeout(timeout + _IO_TIMEOUT_SECONDS)
        try:
            result = _dispatch(self.agent_id, args, timeout)
        except Exception as exc:
            _reply(conn, {"error": str(exc)[:300]})
            return
        output = result.output
        if len(output.encode("utf-8", "ignore")) > _MAX_OUTPUT_BYTES:
            output = output[:_MAX_OUTPUT_BYTES]
        _reply(
            conn,
            {
                "output": output,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
        )


def _dispatch(agent_id: str, args: list[str], timeout: int):
    from agent.credential_broker import (
        request_lark_auth_complete,
        request_lark_auth_start,
        request_lark_cli,
    )

    if len(args) >= 2 and args[0] == "auth" and args[1] == "login":
        # Device-code OAuth rides the dedicated purposes: argv is rebuilt
        # server-side, so shape drift here can never block authorization.
        if "--device-code" in args:
            index = args.index("--device-code")
            code = args[index + 1] if index + 1 < len(args) else ""
            return request_lark_auth_complete(
                agent_id, device_code=code, timeout_seconds=timeout
            )
        scope = ""
        if "--scope" in args:
            index = args.index("--scope")
            scope = args[index + 1] if index + 1 < len(args) else ""
        return request_lark_auth_start(
            agent_id, scope=scope, timeout_seconds=timeout
        )
    return request_lark_cli(agent_id, args, timeout_seconds=timeout)


def _reply(conn: socket.socket, body: dict) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(min(65536, size - len(data)))
        if not chunk:
            raise OSError("lark relay connection closed early")
        data += chunk
    return data


def _ensure_dirs() -> tuple[Path, Path]:
    RELAY_ROOT.mkdir(mode=0o711, exist_ok=True)
    os.chmod(RELAY_ROOT, 0o711)
    bin_dir = RELAY_ROOT / "bin"
    bin_dir.mkdir(mode=0o755, exist_ok=True)
    os.chmod(bin_dir, 0o755)
    return RELAY_ROOT, bin_dir


def _ensure_shim(bin_dir: Path) -> Path:
    shim = bin_dir / "lark-cli"
    current = None
    try:
        current = shim.read_text(encoding="utf-8")
    except OSError:
        pass
    if current != _SHIM_SOURCE:
        tmp = shim.with_suffix(".tmp")
        tmp.write_text(_SHIM_SOURCE, encoding="utf-8")
        os.chmod(tmp, 0o755)
        os.replace(tmp, shim)
    os.chmod(shim, 0o755)
    return shim


def ensure_relay_env(agent_id: str, uid: int, env: dict[str, str]) -> None:
    """Start (or reuse) the per-agent relay and wire PATH + socket env vars.

    Failure must never take the terminal down — callers treat any raise as
    "run without the shim", which degrades to the pre-relay behaviour.
    """

    _, bin_dir = _ensure_dirs()
    _ensure_shim(bin_dir)
    with _relays_lock:
        relay = _relays.get(uid)
        if relay is None or relay.agent_id != agent_id or not relay.alive():
            relay = _Relay(agent_id, uid, RELAY_ROOT / f"{uid}.sock")
            relay.start()
            _relays[uid] = relay
    env[SOCKET_ENV] = str(relay.socket_path)
    path_text = env.get("PATH", "") or os.environ.get("PATH", "")
    bin_text = str(bin_dir)
    parts = [p for p in path_text.split(":") if p]
    if not parts or parts[0] != bin_text:
        env["PATH"] = ":".join([bin_text] + [p for p in parts if p != bin_text])
