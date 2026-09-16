"""Exercise real request construction and post-tool pressure, not a copied formula."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from tests.run_agent.test_post_tool_compression_attempt_cap import (
    agent, _tool_response, _stop_response,
)


@pytest.mark.parametrize('with_usage', [False, True])
@pytest.mark.parametrize('echo', [False, True])
def test_long_think_short_tool_does_not_cause_false_compaction(agent, echo, with_usage):
    with patch('agent.context_compressor.get_model_context_length', return_value=200000):
        compressor = ContextCompressor(model='test/model', quiet_mode=True)
    # One 95k-character trace fits; its duplicate crosses this threshold.
    compressor.threshold_tokens = 40000
    compressor.tail_token_budget = 8000
    agent.context_compressor = compressor
    trace = 'Consider the next tool result. ' * 3200
    reply = _tool_response(0)
    reply.choices[0].message.reasoning_content = trace
    sent = []
    responses = iter([reply, _stop_response()])

    def respond(**kwargs):
        sent.append(deepcopy(kwargs['messages']))
        response = next(responses)
        if with_usage:
            response.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
        return response

    agent.client.chat.completions.create.side_effect = respond
    with (
        patch.object(agent, '_needs_thinking_reasoning_pad', return_value=echo),
        patch.object(agent, '_compress_context', side_effect=AssertionError('false compression')),
        patch.object(agent, '_persist_session'),
        patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
        patch('run_agent.handle_function_call', return_value='{"ok":true}'),
    ):
        result = agent.run_conversation('Inspect the device, then continue the original task.')
    assert result['completed'] is True
    assert len(sent) == 2
    tool_turn = next(m for m in sent[1] if m.get('tool_calls'))
    assert 'reasoning' not in tool_turn
    if echo:
        assert tool_turn['reasoning_content'] == trace
    else:
        assert 'reasoning_content' not in tool_turn
    # The display/storage copy survives regardless of provider replay policy.
    assert any(m.get('reasoning') == trace for m in result['messages'])

    if with_usage:
        assert compressor.last_real_prompt_tokens == 1000
        assert compressor.last_rough_tokens_when_real_prompt_fit > 0
        assert compressor.compression_count == 0


@pytest.mark.parametrize('runtime_window', [None, 272000])
def test_explicit_runtime_window_survives_a_different_configured_route(runtime_window):
    """A session override must not be discarded with the default model's config."""
    from run_agent import AIAgent

    cfg = {'model': {'default': 'gpt-5.6-sol', 'provider': 'custom',
                     'base_url': 'https://old.invalid/v1', 'context_length': 128000},
           'compression': {}, 'agent': {}}
    with (
        patch('hermes_cli.config.load_config', return_value=cfg),
        patch('hermes_cli.config.load_config_readonly', return_value=cfg),
        patch('run_agent.OpenAI'),
        patch('agent.context_compressor.get_model_context_length',
              side_effect=lambda *args, **kw: kw.get('config_context_length') or 1050000),
    ):
        instance = AIAgent(model='gpt-5.6-sol', provider='custom',
                           base_url='https://active.invalid/v1', api_key='test-only-key',
                           config_context_length=runtime_window,
                           skip_tool_loading=True, skip_context_files=True,
                           skip_memory=True, quiet_mode=True)
        assert instance._config_context_length == runtime_window
        assert instance.context_compressor.context_length == (runtime_window or 1050000)
        assert instance.context_compressor.threshold_tokens == 244800


@pytest.mark.parametrize('usage', [None, 50000])
def test_large_tool_growth_never_promotes_characters_to_tokens(agent, usage):
    comp = ContextCompressor(model='test/model', config_context_length=272000, quiet_mode=True)
    agent.context_compressor = comp
    sent = []
    responses = iter([_tool_response(i) for i in range(4)] + [_stop_response()])
    def respond(**kwargs):
        sent.append(deepcopy(kwargs['messages']))
        result = next(responses)
        if usage:
            result.usage = SimpleNamespace(prompt_tokens=usage, input_tokens=0,
                completion_tokens=90000, total_tokens=usage + 90000,
                prompt_tokens_details=SimpleNamespace(cached_tokens=49000),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=89900))
        return result
    agent.client.chat.completions.create.side_effect = respond
    with (
        patch.object(agent, '_compress_context', side_effect=AssertionError('false compaction')),
        patch.object(agent, '_persist_session'), patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
        patch('run_agent.handle_function_call', return_value='资料' * 31250),
    ):
        result = agent.run_conversation('GOAL-480: read the document and preserve requirements.')
    assert result['completed']
    assert len(sent) == 5
    assert comp.automatic_compaction_tokens == (usage or 0)
    assert any('GOAL-480' in str(m.get('content')) for m in sent[-1])
    assert sum(len(str(m.get('content', ''))) for m in sent[-1] if m['role'] == 'tool') >= 250000


def test_measured_threshold_still_compacts_once_then_waits_for_usage(agent):
    comp = ContextCompressor(model='test/model', config_context_length=272000, quiet_mode=True)
    agent.context_compressor = comp
    first = _tool_response(0)
    first.usage = SimpleNamespace(prompt_tokens=250000, completion_tokens=20, total_tokens=250020)
    agent.client.chat.completions.create.side_effect = [first, _stop_response()]
    def compact(messages, system_message, **kwargs):
        comp.last_prompt_tokens = -1
        comp.awaiting_real_usage_after_compression = True
        return messages, system_message
    with (
        patch.object(agent, '_compress_context', side_effect=compact) as compress,
        patch.object(agent, '_persist_session'), patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
        patch('run_agent.handle_function_call', return_value='observed'),
    ):
        result = agent.run_conversation('Inspect and report.')
    assert result['completed']
    assert compress.call_count == 1
    assert comp.automatic_compaction_tokens == 0


@pytest.mark.parametrize('explicit_overflow', [False, True])
def test_large_history_generic_error_is_not_overflow_but_explicit_error_recovers(agent, explicit_overflow):
    comp = ContextCompressor(model='test/model', config_context_length=272000, quiet_mode=True)
    agent.context_compressor = comp
    err = Exception('context length exceeded' if explicit_overflow else 'error')
    err.status_code = 400
    agent.client.chat.completions.create.side_effect = [err, _stop_response()]
    history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': '资料' * 2000}
               for i in range(100)]
    def compact(messages, system_message, **kwargs):
        comp.last_prompt_tokens = -1
        comp.awaiting_real_usage_after_compression = True
        return [{'role': 'user', 'content': 'Original goal: inspect and report.'}], system_message
    with (
        patch.object(agent, '_compress_context', side_effect=compact) as compress,
        patch.object(agent, '_persist_session') as persist,
        patch.object(agent, '_save_trajectory'), patch.object(agent, '_cleanup_task_resources'),
        patch('agent.conversation_loop.time.sleep'),
    ):
        result = agent.run_conversation('Inspect and report.', conversation_history=history)
    assert compress.call_count == int(explicit_overflow)
    assert result['completed'] == explicit_overflow
    assert persist.called
