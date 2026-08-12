"""lark-cli sandbox relay: frame protocol, auth mapping, peer gating."""

import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.credential_broker import LarkCLIResult  # noqa: E402
from tools import lark_cli_relay  # noqa: E402


def _roundtrip(path: str, body: dict) -> dict:
    payload = json.dumps(body).encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(path)
        client.sendall(struct.pack(">I", len(payload)) + payload)
        head = client.recv(4)
        size = struct.unpack(">I", head)[0]
        data = b""
        while len(data) < size:
            data += client.recv(size - len(data))
    return json.loads(data)


class LarkCLIRelayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        self.relay = lark_cli_relay._Relay(
            "agent-a", os.getuid(), Path(self.tmp.name) / "relay.sock"
        )
        self.relay.start()

        def fake_cli(agent_id, args, *, timeout_seconds, socket_path=None):
            self.calls.append(("cli", agent_id, list(args), timeout_seconds))
            return LarkCLIResult(output='{"ok":true}', exit_code=0)

        def fake_start(agent_id, *, scope="", timeout_seconds, socket_path=None):
            self.calls.append(("start", agent_id, scope, timeout_seconds))
            return LarkCLIResult(output='{"verification_url":"u"}', exit_code=0)

        def fake_complete(
            agent_id, *, device_code, timeout_seconds, socket_path=None
        ):
            self.calls.append(("complete", agent_id, device_code, timeout_seconds))
            return LarkCLIResult(output='{"ok":true}', exit_code=0)

        patcher = mock.patch.multiple(
            "agent.credential_broker",
            request_lark_cli=fake_cli,
            request_lark_auth_start=fake_start,
            request_lark_auth_complete=fake_complete,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_business_args_ride_the_lark_cli_purpose(self):
        reply = _roundtrip(
            str(self.relay.socket_path),
            {"args": ["mail", "+triage", "--page-size", "1"], "timeout_seconds": 30},
        )
        self.assertEqual(reply["exit_code"], 0)
        self.assertEqual(
            self.calls, [("cli", "agent-a", ["mail", "+triage", "--page-size", "1"], 30)]
        )

    def test_auth_login_forms_map_to_dedicated_purposes(self):
        _roundtrip(
            str(self.relay.socket_path),
            {"args": ["auth", "login", "--no-wait", "--json", "--scope", "mail:x"]},
        )
        _roundtrip(
            str(self.relay.socket_path),
            {"args": ["auth", "login", "--device-code", "abcdef123456", "--json"]},
        )
        kinds = [(c[0], c[2]) for c in self.calls]
        self.assertEqual(
            kinds, [("start", "mail:x"), ("complete", "abcdef123456")]
        )

    def test_invalid_frames_and_broker_errors_return_error_payloads(self):
        reply = _roundtrip(str(self.relay.socket_path), {"args": "not-a-list"})
        self.assertIn("error", reply)
        with mock.patch(
            "agent.credential_broker.request_lark_cli",
            side_effect=RuntimeError("lark-cli is busy; retry shortly"),
        ):
            reply = _roundtrip(
                str(self.relay.socket_path), {"args": ["mail", "+triage"]}
            )
        self.assertEqual(reply["error"], "lark-cli is busy; retry shortly")

    def test_wrong_peer_uid_is_rejected(self):
        stranger = lark_cli_relay._Relay(
            "agent-a", os.getuid() + 1, Path(self.tmp.name) / "stranger.sock"
        )
        stranger.start()
        reply = _roundtrip(
            str(stranger.socket_path), {"args": ["mail", "+triage"]}
        )
        self.assertIn("not authorized", reply["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
