from gateway.session_context import clear_turn_vars, set_turn_vars
from hermes_cli import middleware


def test_silent_automation_skips_llm_middleware(monkeypatch):
    request = {"messages": [{"role": "user", "content": "frozen"}]}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("silent turn reached generic middleware")

    monkeypatch.setattr(middleware, "_has_middleware", lambda _kind: True)
    monkeypatch.setattr(middleware, "_get_middleware_callbacks", unexpected)
    monkeypatch.setattr(middleware, "_invoke_middleware", unexpected)

    tokens = set_turn_vars(
        turn_id="silent-middleware-turn",
        execution_policy="silent_automation",
    )
    try:
        llm_result = middleware.apply_llm_request_middleware(request)
        assert llm_result.payload is request
        assert llm_result.original_payload is request
        assert llm_result.changed is False

        assert (
            middleware.run_llm_execution_middleware(
                request,
                lambda payload: ("llm", payload),
            )
            == ("llm", request)
        )
    finally:
        clear_turn_vars(tokens)


def test_silent_automation_skips_tool_request_and_execution_middleware(monkeypatch):
    args = {"command": "trusted-helper"}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("silent turn reached tool middleware or relay")

    monkeypatch.setattr(middleware, "_has_middleware", unexpected)
    monkeypatch.setattr(middleware, "_get_middleware_callbacks", unexpected)
    from agent import relay_runtime

    monkeypatch.setattr(relay_runtime, "apply_tool_request_intercepts", unexpected)

    tokens = set_turn_vars(
        turn_id="silent-tool-middleware-turn",
        execution_policy="silent_automation",
    )
    try:
        request_result = middleware.apply_tool_request_middleware(
            "terminal", args, session_id="session-1"
        )
        assert request_result.payload is args
        assert request_result.original_payload is args
        assert request_result.changed is False
        assert request_result.trace == []

        assert (
            middleware.run_tool_execution_middleware(
                "terminal", args, lambda payload: ("tool", payload)
            )
            == ("tool", args)
        )
    finally:
        clear_turn_vars(tokens)
