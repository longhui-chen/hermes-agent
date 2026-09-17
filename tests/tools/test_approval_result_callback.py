"""The optional result observer never changes the approval decision."""
import uuid
import pytest
from tools import approval, approval_gateway_wait as wait


@pytest.mark.parametrize('choice,state', [('once', 'set'), ('deny', 'set'), (None, 'timeout')])
@pytest.mark.parametrize('raises', [False, True])
def test_result_callback(choice, state, raises, monkeypatch):
    session = uuid.uuid4().hex
    observed = []
    monkeypatch.setattr(wait._ctx, '_fire_approval_hook', lambda *a, **kw: None)
    monkeypatch.setattr(wait, '_poll_event', lambda *a, **kw: state)
    def cleanup():
        raise AssertionError('result-aware callback takes precedence')
    def on_result(**result):
        observed.append(result)
        if raises:
            raise RuntimeError('observer failed')
    cleanup.on_result = on_result
    def notify(data):
        approval._gateway_queues[session][0].result = choice
        return cleanup
    result = wait._await_gateway_decision(session, notify, {'command': 'echo safe'})
    assert result['choice'] == choice
    assert result['resolved'] is (state == 'set')
    assert observed == [{'resolved': state == 'set', 'choice': choice, 'reason': None}]
    assert session not in approval._gateway_queues
