"""Hermetic HTTP flow coverage for the device meeting adapter."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from tests.tools._profile_scope import mux_profile_scope
from tools.device_meetings_tool import device_meetings_tool


def test_device_meetings_local_server_envelope_flow(monkeypatch):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler contract
            seen["path"] = self.path
            seen["token"] = self.headers.get("X-Zettlab-Agent-Action-Token")
            body = json.dumps(
                {"code": 200, "data": {"meetings": [{"id": "meeting-flow"}]}},
                separators=(",", ":"),
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scope = {
            "ZET_CHAT_APPEND_URL": f"http://127.0.0.1:{server.server_port}/chat/append",
            "ZETTLAB_AGENT_ACTION_TOKEN": "flow-token",
        }
        with mux_profile_scope(monkeypatch, scope):
            result = device_meetings_tool({"action": "list", "limit": 20, "offset": 0})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert json.loads(result) == {"meetings": [{"id": "meeting-flow"}]}
    parsed = urlparse(seen["path"])
    assert parsed.path == "/api/v1/internal/meetings"
    assert parsed.query == "limit=20&offset=0"
    assert seen["token"] == "flow-token"
