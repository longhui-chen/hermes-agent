from types import SimpleNamespace

from run_agent import AIAgent


def _agent(callback):
    recorded = []
    agent = SimpleNamespace(
        platform="cli",
        _zet_agent_plan_mode_active=False,
        _zet_agent_plan_presented=False,
        _stream_needs_break=False,
        _stream_think_scrubber=None,
        _stream_context_scrubber=None,
        _current_streamed_assistant_text="",
        stream_delta_callback=callback,
        _stream_callback=None,
        _strip_think_blocks=lambda text: text,
        _record_streamed_assistant_text=recorded.append,
    )
    agent._should_suppress_plan_stream_text = (
        lambda: AIAgent._should_suppress_plan_stream_text(agent)
    )
    return agent, recorded


def test_stream_delta_reports_successful_consumer_delivery():
    streamed = []
    agent, recorded = _agent(streamed.append)

    assert AIAgent._fire_stream_delta(agent, "delivered") is True
    assert streamed == ["delivered"]
    assert recorded == ["delivered"]


def test_stream_delta_reports_missing_consumer_delivery():
    agent, recorded = _agent(None)

    assert AIAgent._fire_stream_delta(agent, "not delivered") is False
    assert recorded == []


def test_stream_delta_reports_failed_consumer_delivery():
    def fail(_text):
        raise RuntimeError("display failed")

    agent, recorded = _agent(fail)

    assert AIAgent._fire_stream_delta(agent, "not delivered") is False
    assert recorded == []
