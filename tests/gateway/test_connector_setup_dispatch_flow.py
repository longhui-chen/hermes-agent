"""Exercise actual agent dispatch before the protected clarify callback."""
import json
from types import SimpleNamespace
import pytest
from tests.run_agent.test_tool_call_guardrail_runtime import _make_agent, _mock_tool_call
from agent.agent_runtime_helpers import invoke_tool

@pytest.mark.parametrize('executor', ['sequential', 'concurrent'])
@pytest.mark.parametrize('intent', [
    {'resource_kind': 'camera'},
    {'resource_kind': 'custom_api', 'template_id': 'jira-data-center-pat-api',
     'variables': {'base_url': 'http://192.168.31.33'}},
])
def test_dispatch_preserves_protected_setup_and_returns_only_receipt(executor, intent):
    agent = _make_agent('clarify')
    received = []
    def callback(question, choices, *, connector_setup=None):
        received.append((question, choices, connector_setup))
        return json.dumps({'status': 'submitted'})
    agent.clarify_callback = callback
    args = {'question': 'Connector setup', 'connector_setup': intent}
    if executor == 'sequential':
        messages = []
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content='', tool_calls=[_mock_tool_call('clarify', json.dumps(args))]),
            messages, 'connector-flow',
        )
        result = json.loads(messages[-1]['content'])
    else:
        result = json.loads(invoke_tool(agent, 'clarify', args, 'connector-flow'))
    assert received == [('Connector setup', None, intent)]
    assert result['status'] == 'submitted'
    assert 'user_response' not in result
