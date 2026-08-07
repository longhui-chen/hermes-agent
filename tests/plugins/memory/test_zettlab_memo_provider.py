import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from plugins.memory.zettlab_memo import ZettlabMemoProvider


class _Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, self.headers.get("X-Zettlab-Agent-Action-Token"), body))
        payload = {"memories": [{"statement": "用户偏好简洁回答"}]} if self.path.endswith("/prefetch") else {"ok": True}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def test_provider_prefetch_and_native_write(monkeypatch, tmp_path):
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/api/v1/internal/memory-provider/zettlab-memo"
    monkeypatch.setenv("ZETTLAB_MEMO_PROVIDER_URL", endpoint)
    monkeypatch.setenv("ZETTLAB_MEMO_ACTION_TOKEN", "secret")
    try:
        provider = ZettlabMemoProvider()
        provider.initialize("session-1", hermes_home=str(tmp_path), agent_identity="main", user_id="user-1")
        assert "用户偏好简洁回答" in provider.prefetch("用户喜欢什么？")
        provider.on_memory_write("add", "user", "用户偏好简洁回答", {"session_id": "session-1"})
        assert [item[0].rsplit("/", 1)[-1] for item in _Handler.requests] == ["prefetch", "native-write"]
        assert all(item[1] == "secret" for item in _Handler.requests)
        assert _Handler.requests[1][2]["account_id"] == "user-1"
    finally:
        server.shutdown()
        server.server_close()
