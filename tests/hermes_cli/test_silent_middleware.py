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
