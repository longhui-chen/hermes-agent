import contextvars


def test_runtime_auxiliary_task_config_overrides_profile_config(monkeypatch):
    from agent import auxiliary_client as mod

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "auxiliary": {
                "vision": {
                    "provider": "custom",
                    "model": "profile-vision",
                    "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                }
            }
        },
    )

    mod.clear_runtime_main()
    try:
        mod.set_runtime_auxiliary_task_configs({"vision": {}})
        assert mod._get_auxiliary_task_config("vision") == {}

        mod.set_runtime_auxiliary_task_configs(
            {"vision": {"provider": "custom", "model": "session-vision"}}
        )
        assert mod._get_auxiliary_task_config("vision") == {
            "provider": "custom",
            "model": "session-vision",
        }
    finally:
        mod.clear_runtime_main()


def test_runtime_auxiliary_task_config_is_context_scoped():
    from agent import auxiliary_client as mod

    ctx_a = contextvars.Context()
    ctx_b = contextvars.Context()

    ctx_a.run(
        mod.set_runtime_auxiliary_task_configs,
        {"vision": {"provider": "custom", "model": "session-a"}},
    )
    ctx_b.run(
        mod.set_runtime_auxiliary_task_configs,
        {"vision": {"provider": "custom", "model": "session-b"}},
    )

    assert ctx_a.run(mod._get_auxiliary_task_config, "vision")["model"] == "session-a"
    assert ctx_b.run(mod._get_auxiliary_task_config, "vision")["model"] == "session-b"
