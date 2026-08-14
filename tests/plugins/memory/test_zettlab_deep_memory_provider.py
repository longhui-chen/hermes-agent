import json
import logging
import sqlite3
import threading
import time

from plugins.memory.zettlab_deep_memory import (
    DeepMemoryMCPToolError,
    ZettlabDeepMemoryProvider,
    register,
)


def _initialize(provider, tmp_path, **kwargs):
    provider.initialize(
        kwargs.pop("session_id", "session-1"),
        hermes_home=str(tmp_path),
        **kwargs,
    )


def _wait_for(predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_official_register_entry_point_and_env_config_schema():
    class Collector:
        provider = None

        def register_memory_provider(self, provider):
            self.provider = provider

    collector = Collector()
    register(collector)
    assert isinstance(collector.provider, ZettlabDeepMemoryProvider)
    fields = collector.provider.get_config_schema()
    assert {field["env_var"] for field in fields} == {
        "ZETTLAB_DEEP_MEMORY_URL",
        "ZETTLAB_AGENT_ACTION_TOKEN",
    }


def test_model_tool_schemas_exclude_memo_write():
    names = {
        schema["name"] for schema in ZettlabDeepMemoryProvider().get_tool_schemas()
    }
    assert names == {"memo_recall", "memo_confirm", "memo_forget"}


def test_native_memory_is_mirrored_to_deep_memory():
    prompt = ZettlabDeepMemoryProvider().system_prompt_block()

    assert "call the native memory tool once" in prompt
    assert "on_memory_write hook" in prompt
    assert "memo_write is not exposed as a model tool" in prompt


def test_provider_does_not_repurpose_generic_account_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("ZETTLAB_DEEP_MEMORY_URL", "http://127.0.0.1:8400")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    _initialize(provider, tmp_path, user_id="account-1", user_id_alt="user-1")
    try:
        assert provider._user_id == ""
        assert provider._user_id_alt == ""
    finally:
        provider.shutdown()


def test_builtin_memory_addition_is_mirrored_non_blocking(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider,
        tmp_path,
        deep_memory_principal="iam:issuer:user-1",
        deep_memory_subject="user-1",
    )
    captured = {}
    completed = threading.Event()

    def fake_request(endpoint, arguments, *, timeout, trusted):
        if endpoint == "recall":
            return {"items": []}
        captured.update(
            endpoint=endpoint,
            arguments=arguments,
            timeout=timeout,
            trusted=trusted,
        )
        completed.set()
        return {"status": "stored"}

    provider._request = fake_request
    provider.on_turn_start(7, "我喜欢吃苹果")
    provider.on_memory_write(
        "add",
        "user",
        "喜欢吃苹果。",
        {"session_id": "session-1", "tool_call_id": "call-1"},
    )

    assert completed.wait(1.0)
    provider.shutdown()
    assert captured["endpoint"] == "write"
    assert captured["arguments"] == {
        "action": "add",
        "target": "user",
        "content": "喜欢吃苹果。",
        "old_content": "",
        "explicit": False,
    }
    assert captured["trusted"]["turn_id"] == "turn-7"
    assert captured["trusted"]["tool_call_id"].startswith("native-memory:")
    assert captured["trusted"]["source_text"] == "我喜欢吃苹果"


def test_builtin_replace_and_remove_are_mirrored_through_memo_write(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    calls = []
    completed = threading.Event()

    def fake_request(endpoint, arguments, *, timeout, trusted):
        calls.append((endpoint, arguments))
        if len(calls) == 2:
            completed.set()
        return {"status": "stored", "action": "native_" + arguments["action"]}

    provider._request = fake_request
    provider.on_memory_write(
        "replace",
        "user",
        "喜欢吃梨。",
        {"old_text": "喜欢吃苹果。", "turn_id": "turn-1"},
    )
    provider.on_memory_write(
        "remove",
        "user",
        "",
        {"old_text": "喜欢吃梨。", "turn_id": "turn-1"},
    )

    assert completed.wait(1.0)
    provider.shutdown()
    assert {call[1]["action"] for call in calls} == {"replace", "remove"}
    assert {call[1]["old_content"] for call in calls} == {
        "喜欢吃苹果。",
        "喜欢吃梨。",
    }


def test_prefetch_recall_uses_mcp_and_builds_provider_context(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    captured = []
    started = threading.Event()

    def fake_request(endpoint, arguments, *, timeout, trusted):
        captured.append({
            "endpoint": endpoint,
            "arguments": arguments,
            "trusted": trusted,
        })
        started.set()
        return {"items": [{"statement": "喜欢吃苹果", "predicate": "prefers"}]}

    provider._request = fake_request
    provider.on_turn_start(1, "用户喜欢什么")
    assert started.wait(1.0)
    context = json.loads(provider.prefetch("用户喜欢什么", session_id="session-1"))

    assert len(captured) == 1
    assert captured[0]["endpoint"] == "recall"
    assert captured[0]["arguments"] == {"query": "用户喜欢什么", "limit": 8}
    assert captured[0]["trusted"]["tool_call_id"] == "prefetch:turn-1"
    assert context["kind"] == "zettlab_deep_memory_data"
    assert context["facts"][0]["predicate"] == "prefers"
    provider.shutdown()


def test_on_turn_start_prefetch_is_non_blocking(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    _initialize(provider, tmp_path, deep_memory_principal="user-1")
    started = threading.Event()
    release = threading.Event()

    def fake_request(*_args, **_kwargs):
        started.set()
        release.wait(1.0)
        return {"items": []}

    provider._request = fake_request
    provider.on_turn_start(1, "需要召回的问题")
    assert started.wait(0.25)
    assert provider._prefetch_thread is not None
    assert provider._prefetch_thread.is_alive()
    release.set()
    provider._prefetch_thread.join(timeout=1.0)
    provider.shutdown()


def test_provider_rejects_non_loopback_url(monkeypatch):
    monkeypatch.setenv("ZETTLAB_DEEP_MEMORY_URL", "https://example.com/memory")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    try:
        provider.initialize("session-1", deep_memory_principal="user-1")
    except RuntimeError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback deep-memory URL was accepted")


def test_provider_rejects_direct_model_memo_write(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    provider = ZettlabDeepMemoryProvider()
    _initialize(provider, tmp_path, deep_memory_principal="user-1")
    called = False

    def fake_request(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"status": "stored"}

    provider._request = fake_request
    result = json.loads(
        provider.handle_tool_call(
            "memo_write",
            {"action": "add", "target": "user", "content": "fact"},
            user_id="user-1",
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="call-1",
        )
    )

    assert "error" in result
    assert "Unknown deep memory tool" in result["error"]
    assert called is False
    provider.shutdown()


def test_tool_calls_use_authenticated_mcp_transport(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider,
        tmp_path,
        deep_memory_principal="iam:issuer:user-1",
        deep_memory_subject="user-1",
    )
    captured = {}
    completed = threading.Event()

    class Response:
        status = 200

        def __init__(self, structured):
            self.structured = structured

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({
                "jsonrpc": "2.0",
                "id": "call-1",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(self.structured),
                        }
                    ],
                    "structuredContent": self.structured,
                    "isError": False,
                },
            }).encode()

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        if body["params"]["name"] == "memo_recall":
            return Response({"items": []})
        captured["url"] = request.full_url
        captured["body"] = body
        captured["token"] = request.headers["X-zettlab-agent-action-token"]
        captured["timeout"] = timeout
        completed.set()
        return Response({"status": "stored"})

    monkeypatch.setattr(
        "plugins.memory.zettlab_deep_memory.urllib.request.urlopen",
        fake_urlopen,
    )
    provider.on_turn_start(1, "I prefer concise answers.")
    provider.on_memory_write(
        "add",
        "user",
        "I prefer concise answers.",
        {"session_id": "session-1", "tool_call_id": "call-1"},
    )

    assert completed.wait(1.0)
    provider.shutdown()
    assert captured["url"].endswith("/deep-memory/mcp")
    assert captured["token"] == "action-token"
    assert captured["body"]["method"] == "tools/call"
    assert captured["body"]["params"]["name"] == "memo_write"
    runtime = captured["body"]["params"]["_meta"]["zettlab/runtime"]
    assert runtime["user_id"] == "iam:issuer:user-1"
    assert runtime["source_text"] == "I prefer concise answers."


def test_mcp_tool_error_is_not_treated_as_a_success(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider,
        tmp_path,
        deep_memory_principal="iam:issuer:user-1",
        deep_memory_subject="user-1",
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({
                "jsonrpc": "2.0",
                "id": "call-1",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error_type": "validation"}),
                        }
                    ],
                    "structuredContent": {"error_type": "validation"},
                    "isError": True,
                },
            }).encode()

    monkeypatch.setattr(
        "plugins.memory.zettlab_deep_memory.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    with caplog.at_level(logging.DEBUG, logger="plugins.memory.zettlab_deep_memory"):
        result = json.loads(
            provider.handle_tool_call(
                "memo_recall",
                {"query": "where do I live?", "limit": 8},
                turn_id="turn-1",
                tool_call_id="call-1",
            )
        )

    assert result == {"error": "Deep Memory is temporarily unavailable"}
    assert "returned an error (validation)" in caplog.text
    provider.shutdown()


def test_transient_memo_write_failure_retries_with_stable_id(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    provider._mirror_retry_delay = lambda _attempt: 0.01
    _initialize(
        provider, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    calls = []
    completed = threading.Event()

    def fake_request(endpoint, arguments, *, timeout, trusted):
        calls.append((endpoint, dict(arguments), dict(trusted)))
        if len(calls) == 1:
            raise DeepMemoryMCPToolError("unavailable")
        completed.set()
        return {"status": "stored"}

    provider._request = fake_request
    provider.on_memory_write(
        "add",
        "user",
        "爸爸喜欢吃苹果。",
        {"turn_id": "turn-8", "tool_call_id": "native-call-8"},
    )

    assert completed.wait(1.0)
    _wait_for(lambda: provider._mirror_outbox.counts()["pending"] == 0)
    provider.shutdown()
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "write"
    assert (
        calls[0][1]
        == calls[1][1]
        == {
            "action": "add",
            "target": "user",
            "content": "爸爸喜欢吃苹果。",
            "old_content": "",
            "explicit": False,
        }
    )
    assert calls[0][2]["tool_call_id"] == calls[1][2]["tool_call_id"]
    assert calls[0][2]["tool_call_id"].startswith("native-memory:")


def test_pending_memo_write_is_recovered_after_provider_restart(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    first = ZettlabDeepMemoryProvider()
    first._mirror_retry_delay = lambda _attempt: 0.5
    failed = threading.Event()

    def fail_request(endpoint, arguments, *, timeout, trusted):
        failed.set()
        raise TimeoutError("auxiliary model timed out")

    first._request = fail_request
    _initialize(
        first, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    first.on_memory_write(
        "replace",
        "user",
        "喜欢吃橘子。",
        {
            "old_text": "喜欢吃西瓜。",
            "turn_id": "turn-9",
            "tool_call_id": "native-call-9",
        },
    )
    assert failed.wait(1.0)
    outbox_path = first._mirror_outbox.path

    def retry_was_committed():
        with sqlite3.connect(outbox_path) as connection:
            row = connection.execute(
                "SELECT attempt_count FROM mirror_outbox WHERE state = 'pending'"
            ).fetchone()
        return row is not None and row[0] == 1

    _wait_for(retry_was_committed)
    with sqlite3.connect(outbox_path) as connection:
        original_id = connection.execute(
            "SELECT id FROM mirror_outbox WHERE state = 'pending'"
        ).fetchone()[0]
    first.shutdown()

    delivered = threading.Event()
    recovered_calls = []
    second = ZettlabDeepMemoryProvider()

    def succeed_request(endpoint, arguments, *, timeout, trusted):
        recovered_calls.append((endpoint, dict(arguments), dict(trusted)))
        delivered.set()
        return {"status": "stored"}

    second._request = succeed_request
    _initialize(
        second, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    assert delivered.wait(1.5)
    _wait_for(lambda: second._mirror_outbox.counts()["pending"] == 0)
    second.shutdown()
    assert len(recovered_calls) == 1
    assert recovered_calls[0][0] == "write"
    assert recovered_calls[0][1] == {
        "action": "replace",
        "target": "user",
        "content": "喜欢吃橘子。",
        "old_content": "喜欢吃西瓜。",
        "explicit": False,
    }
    assert recovered_calls[0][2]["tool_call_id"] == (f"native-memory:{original_id}")


def test_structured_validation_failure_dead_letters_without_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    provider._mirror_retry_delay = lambda _attempt: 0.01
    calls = []

    def fail_validation(endpoint, arguments, *, timeout, trusted):
        calls.append((endpoint, dict(arguments), dict(trusted)))
        raise DeepMemoryMCPToolError("validation")

    provider._request = fail_validation
    _initialize(
        provider, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )
    provider.on_memory_write(
        "add",
        "user",
        "Frank 是我的老板，也喜欢看《七龙珠》。",
        {"turn_id": "turn-10"},
    )

    _wait_for(lambda: provider._mirror_outbox.counts()["dead"] == 1)
    provider.shutdown()
    assert len(calls) == 3
    assert {call[0] for call in calls} == {"write"}
    assert {call[1]["action"] for call in calls} == {"add"}
    assert {call[1]["content"] for call in calls} == {
        "Frank 是我的老板，也喜欢看《七龙珠》。"
    }
    assert len({call[2]["tool_call_id"] for call in calls}) == 1


def test_outbox_avoids_wal_on_managed_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "ZETTLAB_DEEP_MEMORY_URL",
        "http://127.0.0.1:9090/api/v1/internal/deep-memory",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "secret")
    provider = ZettlabDeepMemoryProvider()
    _initialize(
        provider, tmp_path, deep_memory_principal="user-1", deep_memory_subject="user-1"
    )

    with sqlite3.connect(provider._mirror_outbox.path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]

    provider.shutdown()
    assert journal_mode == "delete"
    assert synchronous == 2
