"""Long reasoning must not manufacture context pressure or failed checkpoints."""
from copy import deepcopy
from functools import partial
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.message_sanitization import apply_reasoning_content_policy
from agent.model_metadata import estimate_messages_tokens_rough, project_messages_for_token_estimate


@pytest.mark.parametrize('estimate', [estimate_messages_tokens_rough, lambda ms: sum(map(_estimate_msg_budget_tokens, ms))])
def test_duplicate_trace_has_single_budget(estimate):
    trace = '检查工具反馈，保留用户要求。' * 7000
    single = {'role': 'assistant', 'content': '', 'reasoning_content': trace}
    double = dict(single, reasoning=trace)
    assert estimate([single]) == estimate([double])


@pytest.mark.parametrize('echo', [True, False])
def test_estimate_follows_real_replay_policy_without_mutating_history(echo):
    trace = 'Check the result before proceeding. ' * 3000
    messages = [{'role': 'assistant', 'content': '', 'reasoning': trace,
                 'reasoning_content': trace, 'tool_calls': [{'id': 'c1', 'type': 'function',
                 'function': {'name': 'terminal', 'arguments': '{"command":"pwd"}'}}]}]
    before = deepcopy(messages)
    projected = project_messages_for_token_estimate(
        messages, partial(apply_reasoning_content_policy, needs_thinking_pad=echo))
    assert messages == before
    assert 'reasoning' not in projected[0]
    if echo:
        assert projected[0]['reasoning_content'] == trace
        assert estimate_messages_tokens_rough(projected) > 20000
    else:
        assert 'reasoning_content' not in projected[0]
        assert estimate_messages_tokens_rough(projected) < 200


def test_automatic_expanding_summary_is_not_committed():
    with patch('agent.context_compressor.get_model_context_length', return_value=200000):
        comp = ContextCompressor(model='test/model', protect_first_n=1,
                                 protect_last_n=1, quiet_mode=True)
    comp.tail_token_budget = 10
    messages = [{'role': 'system', 'content': 'Keep original constraints.'}]
    for i in range(12):
        messages.extend([{'role': 'user', 'content': f'requirement {i}'},
                         {'role': 'assistant', 'content': f'verified result {i}'}])
    before = deepcopy(messages)
    with patch.object(comp, '_generate_summary', return_value='expanded speculation ' * 10000):
        result = comp.compress(messages, current_tokens=160000)
    assert result == before
    assert messages == before
    assert comp.compression_count == 0
    assert not comp._last_compression_made_progress
    assert comp._ineffective_compression_count == 1
    assert comp._previous_summary is None


def test_successful_usage_calibrates_without_compaction_and_checks_do_not_drift():
    import random

    with patch('agent.context_compressor.get_model_context_length', return_value=200000):
        comp = ContextCompressor(model='test/model', quiet_mode=True)
    comp.context_length = 200000
    assert comp.threshold_tokens == 150000
    rng = random.Random(17)
    for _ in range(100):
        actual = rng.randrange(100000, 150000)
        rough = rng.randrange(180000, 240000)
        comp.update_from_response({'prompt_tokens': actual,
                                   'request_estimated_tokens': rough})
        for _ in range(10):
            growth = rng.randrange(0, 12000)
            expected = growth <= 7500 and actual + growth < 150000
            assert comp.should_defer_rough_estimate_to_real_usage(rough + growth) == expected
            assert comp.last_rough_tokens_when_real_prompt_fit == rough
    # A provider reading above the trigger overrides an earlier fitting anchor.
    comp.update_from_response({'prompt_tokens': 150001, 'request_estimated_tokens': 200000})
    assert not comp.should_defer_rough_estimate_to_real_usage(200000)
