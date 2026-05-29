import json

from gateway.session_model_overrides import load_session_model_overrides


def test_load_session_model_overrides_filters_runtime_blob(tmp_path):
    path = tmp_path / "session_model_overrides.json"
    path.write_text(
        json.dumps(
            {
                "zettlab:user:a1:42": {
                    "model": "glm-5",
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                    "api_key": "local-ai-proxy",
                    "api_mode": "openai_chat",
                    "context_length": 1000000,
                },
                "missing-model": {"provider": "custom"},
                "wrong-shape": "glm-5",
            }
        ),
        encoding="utf-8",
    )

    assert load_session_model_overrides(path) == {
        "zettlab:user:a1:42": {
            "model": "glm-5",
            "provider": "custom",
            "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
            "api_key": "local-ai-proxy",
            "api_mode": "openai_chat",
            "context_length": 1000000,
        }
    }


def test_load_session_model_overrides_missing_or_invalid(tmp_path):
    assert load_session_model_overrides(tmp_path / "missing.json") == {}

    path = tmp_path / "session_model_overrides.json"
    path.write_text("{broken", encoding="utf-8")
    assert load_session_model_overrides(path) == {}
