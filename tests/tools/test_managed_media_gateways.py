import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from hermes_cli.nous_account import NousPortalAccountInfo


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"


def _load_tool_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, TOOLS_DIR / filename)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_tool_and_agent_modules():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tools"
        or name.startswith("tools.")
        or name == "agent"
        or name.startswith("agent.")
        or name in {"fal_client", "openai"}
    }
    try:
        yield
    finally:
        for name in list(sys.modules):
            if (
                name == "tools"
                or name.startswith("tools.")
                or name == "agent"
                or name.startswith("agent.")
                or name in {"fal_client", "openai"}
            ):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


@pytest.fixture(autouse=True)
def _enable_managed_nous_tools(monkeypatch):
    """Patch the source modules so managed_nous_tools_enabled() returns True
    even after tool modules are dynamically reloaded."""
    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        lambda: NousPortalAccountInfo(
            logged_in=True,
            source="jwt",
            fresh=False,
            paid_service_access=True,
        ),
    )


def _install_fake_tools_package():
    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package
    sys.modules["tools.debug_helpers"] = types.SimpleNamespace(
        DebugSession=lambda *args, **kwargs: types.SimpleNamespace(
            active=False,
            session_id="debug-session",
            log_call=lambda *a, **k: None,
            save=lambda: None,
            get_session_info=lambda: {},
        )
    )
    sys.modules["tools.managed_tool_gateway"] = _load_tool_module(
        "tools.managed_tool_gateway",
        "managed_tool_gateway.py",
    )


def _install_fake_fal_client(captured):
    def submit(model, arguments=None, headers=None):
        raise AssertionError("managed FAL gateway mode should use fal_client.SyncClient")

    class FakeResponse:
        def json(self):
            return {
                "request_id": "req-123",
                "response_url": "http://127.0.0.1:3009/requests/req-123",
                "status_url": "http://127.0.0.1:3009/requests/req-123/status",
                "cancel_url": "http://127.0.0.1:3009/requests/req-123/cancel",
            }

    def _maybe_retry_request(client, method, url, json=None, timeout=None, headers=None):
        captured["submit_via"] = "managed_client"
        captured["http_client"] = client
        captured["method"] = method
        captured["submit_url"] = url
        captured["arguments"] = json
        captured["timeout"] = timeout
        captured["headers"] = headers
        return FakeResponse()

    class SyncRequestHandle:
        def __init__(self, request_id, response_url, status_url, cancel_url, client):
            captured["request_id"] = request_id
            captured["response_url"] = response_url
            captured["status_url"] = status_url
            captured["cancel_url"] = cancel_url
            captured["handle_client"] = client

    class SyncClient:
        def __init__(self, key=None, default_timeout=120.0):
            captured["sync_client_inits"] = captured.get("sync_client_inits", 0) + 1
            captured["client_key"] = key
            captured["client_timeout"] = default_timeout
            self.default_timeout = default_timeout
            self._client = object()

    fal_client_module = types.SimpleNamespace(
        submit=submit,
        SyncClient=SyncClient,
        client=types.SimpleNamespace(
            _maybe_retry_request=_maybe_retry_request,
            _raise_for_status=lambda response: None,
            SyncRequestHandle=SyncRequestHandle,
        ),
    )
    sys.modules["fal_client"] = fal_client_module
    return fal_client_module


def _install_fake_openai_module(
    captured,
    transcription_response=None,
    speech_chunks=(b"fake-audio",),
):
    class FakeSpeechResponse:
        def __enter__(self):
            captured["streaming_response_entered"] = True
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self, chunk_size=None):
            captured["iter_bytes_chunk_size"] = chunk_size
            yield from speech_chunks

        def stream_to_file(self, output_path):
            captured["stream_to_file"] = output_path
            Path(output_path).write_bytes(b"fake-audio")

    class FakeOpenAI:
        def __init__(self, api_key, base_url, **kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["client_kwargs"] = kwargs
            captured["close_calls"] = captured.get("close_calls", 0)

            def create_speech(**kwargs):
                captured["speech_kwargs"] = kwargs
                return FakeSpeechResponse()

            def create_streaming_speech(**kwargs):
                captured["speech_kwargs"] = kwargs
                captured["streaming_create_calls"] = (
                    captured.get("streaming_create_calls", 0) + 1
                )
                return FakeSpeechResponse()

            def create_transcription(**kwargs):
                captured["transcription_kwargs"] = kwargs
                return transcription_response

            self.audio = types.SimpleNamespace(
                speech=types.SimpleNamespace(
                    create=create_speech,
                    with_streaming_response=types.SimpleNamespace(
                        create=create_streaming_speech,
                    ),
                ),
                transcriptions=types.SimpleNamespace(
                    create=create_transcription
                ),
            )

        def close(self):
            captured["close_calls"] += 1

    fake_module = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=Exception,
        APIConnectionError=Exception,
        APITimeoutError=Exception,
        BadRequestError=type("BadRequestError", (Exception,), {}),
    )
    sys.modules["openai"] = fake_module


def test_managed_fal_submit_uses_gateway_origin_and_nous_token(monkeypatch):
    captured = {}
    _install_fake_tools_package()
    _install_fake_fal_client(captured)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("FAL_QUEUE_GATEWAY_URL", "http://127.0.0.1:3009")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    image_generation_tool = _load_tool_module(
        "tools.image_generation_tool",
        "image_generation_tool.py",
    )
    monkeypatch.setattr(image_generation_tool.uuid, "uuid4", lambda: "fal-submit-123")
    
    image_generation_tool._submit_fal_request(
        "fal-ai/flux-2-pro",
        {"prompt": "test prompt", "num_images": 1},
    )

    assert captured["submit_via"] == "managed_client"
    assert captured["client_key"] == "nous-token"
    assert captured["submit_url"] == "http://127.0.0.1:3009/fal-ai/flux-2-pro"
    assert captured["method"] == "POST"
    assert captured["arguments"] == {"prompt": "test prompt", "num_images": 1}
    assert captured["headers"] == {"x-idempotency-key": "fal-submit-123"}
    assert captured["sync_client_inits"] == 1


def test_openai_tts_uses_managed_audio_gateway_when_direct_key_absent(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    monkeypatch.setattr(tts_tool.uuid, "uuid4", lambda: "tts-call-123")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts("hello world", str(output_path), {"openai": {}})

    assert captured["api_key"] == "nous-token"
    assert captured["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert captured["speech_kwargs"]["model"] == "gpt-4o-mini-tts"
    assert captured["speech_kwargs"]["extra_headers"] == {"x-idempotency-key": "tts-call-123"}
    assert captured["streaming_create_calls"] == 1
    assert captured["streaming_response_entered"] is True
    assert captured["iter_bytes_chunk_size"] == 64 * 1024
    assert output_path.read_bytes() == b"fake-audio"
    assert captured["close_calls"] == 1


def test_zettlab_tts_auto_selects_independent_provider(monkeypatch):
    _install_fake_tools_package()
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    assert tts_tool._get_provider({}) == "zettlab"
    assert tts_tool._get_provider({"provider": "edge", "_provider_is_default": True}) == "zettlab"
    assert tts_tool._get_provider({"provider": "edge"}) == "edge"


def test_zettlab_tts_existing_direct_key_selects_openai(monkeypatch):
    _install_fake_tools_package()
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    assert tts_tool._get_provider({}) == "openai"
    assert tts_tool._get_provider({"use_gateway": True}) == "openai"


def test_tts_bounded_file_sink_removes_partial_output(monkeypatch, tmp_path):
    _install_fake_tools_package()
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "oversized.mp3"

    class OversizedResponse:
        headers = {}

        def iter_bytes(self, chunk_size=None):
            del chunk_size
            yield b"1234"
            yield b"5678"

        def close(self):
            return None

    with pytest.raises(RuntimeError, match="exceeds 6 bytes"):
        tts_tool._write_tts_response_to_file(
            OversizedResponse(),
            str(output_path),
            label="test TTS",
            limit=6,
        )

    assert not output_path.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_zettlab_tts_explicit_direct_openai_opt_out_wins(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts(
        "hello world",
        str(output_path),
        {
            "provider": "openai",
            "use_gateway": False,
            "openai": {"model": "tts-1-hd", "voice": "nova", "speed": 1.25},
        },
    )

    assert captured["api_key"] == "direct-openai-key"
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["speech_kwargs"]["model"] == "tts-1-hd"
    assert captured["speech_kwargs"]["voice"] == "nova"
    assert captured["speech_kwargs"]["speed"] == 1.25


def test_zettlab_tts_direct_keys_are_isolated_by_profile(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.setenv("OPENAI_API_KEY", "stale-process-key")
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        profile_a = secret_scope.set_secret_scope(
            {
                "VOICE_TOOLS_OPENAI_KEY": "profile-a-key",
                "OPENAI_BASE_URL": "https://profile-a.example/v1",
            }
        )
        try:
            tts_tool._generate_openai_tts(
                "profile a",
                str(tmp_path / "profile-a.mp3"),
                {"provider": "openai", "use_gateway": False},
            )
            profile_a_key = captured["api_key"]
            profile_a_base_url = captured["base_url"]
        finally:
            secret_scope.reset_secret_scope(profile_a)

        profile_b = secret_scope.set_secret_scope(
            {
                "OPENAI_API_KEY": "profile-b-key",
                "OPENAI_BASE_URL": "https://profile-b.example/v1",
            }
        )
        try:
            tts_tool._generate_openai_tts(
                "profile b",
                str(tmp_path / "profile-b.mp3"),
                {"provider": "openai", "use_gateway": False},
            )
            profile_b_key = captured["api_key"]
            profile_b_base_url = captured["base_url"]
        finally:
            secret_scope.reset_secret_scope(profile_b)
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)

    assert profile_a_key == "profile-a-key"
    assert profile_b_key == "profile-b-key"
    assert profile_a_base_url == "https://profile-a.example/v1"
    assert profile_b_base_url == "https://profile-b.example/v1"


def test_zettlab_tts_direct_opt_out_requires_direct_key(monkeypatch):
    _install_fake_tools_package()
    _install_fake_openai_module({})
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "openai", "use_gateway": False},
    )

    assert tts_tool.check_tts_requirements() is False


def test_openai_tts_forced_gateway_visibility_ignores_stale_direct_key(monkeypatch):
    _install_fake_tools_package()
    _install_fake_openai_module({})
    monkeypatch.setenv("OPENAI_API_KEY", "stale-direct-key")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    monkeypatch.setattr(
        tts_tool,
        "resolve_managed_tool_gateway",
        lambda _capability: None,
    )

    assert tts_tool._has_openai_audio_backend(
        {"provider": "openai", "use_gateway": True}
    ) is False


def test_zettlab_tts_visibility_is_rechecked_across_multiplex_profiles(monkeypatch):
    _install_fake_tools_package()
    _install_fake_openai_module({})
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    from tools import registry as tool_registry

    selected_profile = {"has_backend": False}
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "openai"},
    )
    monkeypatch.setattr(
        tts_tool,
        "_has_openai_audio_backend",
        lambda _config: selected_profile["has_backend"],
    )
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)

    tool_registry._check_fn_cache.clear()
    tool_registry._check_fn_last_good.clear()
    assert tool_registry.registry.get_definitions({"text_to_speech"}) == []

    selected_profile["has_backend"] = True
    definitions = tool_registry.registry.get_definitions({"text_to_speech"})
    assert [item["function"]["name"] for item in definitions] == ["text_to_speech"]


def test_zettlab_tts_preserves_managed_scope_provider(monkeypatch, tmp_path):
    user_home = tmp_path / "user"
    managed_dir = tmp_path / "managed"
    user_home.mkdir()
    managed_dir.mkdir()
    (user_home / "config.yaml").write_text(
        "tts:\n  speed: 1.0\n", encoding="utf-8"
    )
    (managed_dir / "config.yaml").write_text(
        "tts:\n  provider: edge\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(user_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    from hermes_cli import config, managed_scope

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    _install_fake_tools_package()
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")

    tts_config = tts_tool._load_tts_config()
    assert tts_tool._get_provider(tts_config) == "edge"


def test_zettlab_tts_preserves_edge_picker_opt_out(monkeypatch, tmp_path):
    user_home = tmp_path / "user"
    user_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(user_home))
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    from hermes_cli import config, managed_scope, tools_config

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    picker_config = {}
    tools_config.apply_provider_selection(
        "tts",
        "Microsoft Edge TTS",
        picker_config,
    )
    assert picker_config["tts"] == {"provider": "edge", "use_gateway": False}
    config.save_config(picker_config)
    raw_tts = config.read_raw_config()["tts"]
    assert "provider" not in raw_tts
    assert raw_tts["use_gateway"] is False

    _install_fake_tools_package()
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")

    tts_config = tts_tool._load_tts_config()
    assert tts_tool._get_provider(tts_config) == "edge"


def test_zettlab_tts_ignores_openai_gateway_toggle_for_provider_selection(
    monkeypatch,
    tmp_path,
):
    user_home = tmp_path / "user"
    managed_dir = tmp_path / "managed"
    user_home.mkdir()
    managed_dir.mkdir()
    (user_home / "config.yaml").write_text(
        "tts:\n  use_gateway: false\n", encoding="utf-8"
    )
    (managed_dir / "config.yaml").write_text(
        "tts:\n  use_gateway: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(user_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    from hermes_cli import config, managed_scope

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    _install_fake_tools_package()
    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")

    tts_config = tts_tool._load_tts_config()
    assert tts_tool._get_provider(tts_config) == "zettlab"


def test_zettlab_tts_never_sends_action_token_to_custom_endpoint(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-tts.example.test/v1")
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-action-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts(
        "hello world",
        str(output_path),
        {"openai": {"base_url": "https://tts.example.test/v1"}},
    )

    assert captured["api_key"] == "direct-openai-key"
    assert captured["base_url"] == "https://tts.example.test/v1"


def test_openai_tts_use_gateway_overrides_stale_direct_endpoint(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts(
        "hello world",
        str(output_path),
        {
            "use_gateway": True,
            "openai": {
                "api_key": "stale-direct-key",
                "base_url": "https://tts.example.test/v1",
                "model": "tts-1-hd",
            },
        },
    )

    assert captured["api_key"] == "nous-token"
    assert captured["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert captured["speech_kwargs"]["model"] == "gpt-4o-mini-tts"


def test_openai_tts_coerces_direct_only_model_on_managed_gateway(monkeypatch, tmp_path):
    """A tts.openai.model valid only for direct OpenAI (e.g. tts-1-hd) must be
    coerced to a managed-supported model, else the gateway 400s with
    'Unsupported managed OpenAI speech model'."""
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts(
        "hello world", str(output_path), {"openai": {"model": "tts-1-hd"}}
    )

    assert captured["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert captured["speech_kwargs"]["model"] == "gpt-4o-mini-tts"


def test_openai_tts_keeps_direct_only_model_with_direct_key(monkeypatch, tmp_path):
    """With a direct key, the user's tts-1-hd is honored (not coerced)."""
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-direct-key")
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts(
        "hello world", str(output_path), {"openai": {"model": "tts-1-hd"}}
    )

    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["speech_kwargs"]["model"] == "tts-1-hd"


def test_openai_tts_accepts_openai_api_key_as_direct_fallback(monkeypatch, tmp_path):
    captured = {}
    _install_fake_tools_package()
    _install_fake_openai_module(captured)
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-direct-key")
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    tts_tool = _load_tool_module("tools.tts_tool", "tts_tool.py")
    output_path = tmp_path / "speech.mp3"
    tts_tool._generate_openai_tts("hello world", str(output_path), {"openai": {}})

    assert captured["api_key"] == "openai-direct-key"
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["close_calls"] == 1


def test_transcription_uses_model_specific_response_formats(monkeypatch, tmp_path):
    whisper_capture = {}
    _install_fake_tools_package()
    _install_fake_openai_module(whisper_capture, transcription_response="hello from whisper")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "stt:\n  provider: openai\n", encoding="utf-8"
    )
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "nous-token")

    transcription_tools = _load_tool_module(
        "tools.transcription_tools",
        "transcription_tools.py",
    )
    transcription_tools._load_stt_config = lambda: {"provider": "openai"}
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF0000WAVEfmt ")

    whisper_result = transcription_tools.transcribe_audio(str(audio_path), model="whisper-1")
    assert whisper_result["success"] is True
    assert whisper_capture["base_url"] == "https://openai-audio-gateway.nousresearch.com/v1"
    assert whisper_capture["transcription_kwargs"]["response_format"] == "text"
    assert whisper_capture["close_calls"] == 1

    json_capture = {}
    _install_fake_openai_module(
        json_capture,
        transcription_response=types.SimpleNamespace(text="hello from gpt-4o"),
    )
    transcription_tools = _load_tool_module(
        "tools.transcription_tools",
        "transcription_tools.py",
    )

    json_result = transcription_tools.transcribe_audio(
        str(audio_path),
        model="gpt-4o-mini-transcribe",
    )
    assert json_result["success"] is True
    assert json_result["transcript"] == "hello from gpt-4o"
    assert json_capture["transcription_kwargs"]["response_format"] == "json"
    assert json_capture["close_calls"] == 1


PLUGINS_DIR = Path(__file__).resolve().parents[2] / "plugins"


def _load_video_gen_plugin(monkeypatch):
    """Load the FAL video gen plugin in isolation."""
    _install_fake_tools_package()

    # Also need the agent.video_gen_provider ABC
    agent_dir = Path(__file__).resolve().parents[2] / "agent"
    spec = spec_from_file_location(
        "agent.video_gen_provider",
        agent_dir / "video_gen_provider.py",
    )
    assert spec and spec.loader
    mod = module_from_spec(spec)
    sys.modules["agent.video_gen_provider"] = mod
    spec.loader.exec_module(mod)

    # Load the plugin
    plugin_init = PLUGINS_DIR / "video_gen" / "fal" / "__init__.py"
    spec = spec_from_file_location("plugins.video_gen.fal", plugin_init)
    assert spec and spec.loader
    plugin_mod = module_from_spec(spec)
    sys.modules["plugins.video_gen.fal"] = plugin_mod
    spec.loader.exec_module(plugin_mod)
    return plugin_mod


def test_video_gen_happy_horse_uses_alibaba_namespace():
    """Verify the happy-horse family uses alibaba/ not fal-ai/ endpoints."""
    _install_fake_tools_package()

    # Load just the plugin module to check the catalog
    plugin_init = PLUGINS_DIR / "video_gen" / "fal" / "__init__.py"

    agent_dir = Path(__file__).resolve().parents[2] / "agent"
    spec = spec_from_file_location(
        "agent.video_gen_provider",
        agent_dir / "video_gen_provider.py",
    )
    mod = module_from_spec(spec)
    sys.modules["agent.video_gen_provider"] = mod
    spec.loader.exec_module(mod)

    spec = spec_from_file_location("plugins.video_gen.fal", plugin_init)
    plugin_mod = module_from_spec(spec)
    sys.modules["plugins.video_gen.fal"] = plugin_mod
    spec.loader.exec_module(plugin_mod)

    hh = plugin_mod.FAL_FAMILIES["happy-horse"]
    assert hh["text_endpoint"] == "alibaba/happy-horse/text-to-video"
    assert hh["image_endpoint"] == "alibaba/happy-horse/image-to-video"
