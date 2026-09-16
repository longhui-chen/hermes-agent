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
        comp = ContextCompressor(model='test/model', threshold_percent=0.75, quiet_mode=True)
    comp.context_length = 200000
    assert comp.threshold_tokens == 150000
    rng = random.Random(17)
    for _ in range(100):
        actual = rng.randrange(100000, 150000)
        rough = rng.randrange(180000, 240000)
        comp.update_from_response({'prompt_tokens': actual,
                                   'request_estimated_tokens': rough})
        for _ in range(10):
            growth = rng.randrange(0, 2000000)
            expected = True
            assert comp.should_defer_rough_estimate_to_real_usage(rough + growth) == expected
    # A provider reading above the trigger overrides an earlier fitting anchor.
    comp.update_from_response({'prompt_tokens': 150001, 'request_estimated_tokens': 200000})
    assert not comp.should_defer_rough_estimate_to_real_usage(200000)


@pytest.mark.parametrize('window', [64000, 256000, 272000, 1000000])
def test_standard_budget_matches_codex_and_respects_smaller_routes(window):
    from hermes_cli.config import DEFAULT_CONFIG

    comp = ContextCompressor(model='test/model', config_context_length=window,
                             max_tokens=8192, quiet_mode=True)
    ratio = DEFAULT_CONFIG['compression']['threshold']
    assert 0.8 <= ratio < 1
    assert comp.threshold_percent == ratio
    assert comp.threshold_tokens == min(int(window * ratio), 244800, window - 8192)
    assert comp.threshold_tokens <= window - 8192


def test_explicit_lower_ratio_is_not_silently_raised():
    comp = ContextCompressor(model='test/model', config_context_length=256000,
                             threshold_percent=0.5, threshold_tokens_cap=180000,
                             quiet_mode=True)
    assert comp.threshold_tokens == 128000
    assert comp.threshold_percent == 0.5


def test_summary_serializes_observations_not_reasoning_fields():
    comp = ContextCompressor(model='test/model', config_context_length=256000,
                             quiet_mode=True)
    messages = [
        {'role': 'user', 'content': 'Preserve GOAL-480; do not write files.'},
        {'role': 'assistant', 'content': '<think>INLINE_SPECULATION</think>Checking.',
         'reasoning': 'PRIVATE_SPECULATION' * 2000,
         'reasoning_content': 'PRIVATE_SPECULATION' * 2000,
         'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
             'name': 'inspect_fixture', 'arguments': '{"step":1}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'OBSERVED_STEP_1'}]
    serialized = comp._serialize_for_summary(messages)
    assert 'SPECULATION' not in serialized
    assert 'GOAL-480' in serialized
    assert 'OBSERVED_STEP_1' in serialized
    assert 'inspect_fixture' in serialized


def test_display_only_reasoning_cannot_disguise_an_expanding_checkpoint():
    comp = ContextCompressor(model='test/model', config_context_length=256000,
                             protect_first_n=0, protect_last_n=1, quiet_mode=True)
    comp.tail_token_budget = 10
    messages = [{'role': 'user', 'content': 'Continue GOAL-480.'}]
    for i in range(12):
        messages.append({'role': 'assistant', 'content': f'observation {i}',
                         'reasoning_content': 'scratch work ' * 2000})
    messages.append({'role': 'user', 'content': 'Continue.'})
    before = deepcopy(messages)
    policy = partial(apply_reasoning_content_policy, needs_thinking_pad=False)
    def replay_size(items):
        return len(str(project_messages_for_token_estimate(items, policy)).encode("utf-8"))
    with patch.object(comp, '_generate_summary', return_value='expanded checkpoint ' * 100):
        result = comp.compress(messages, current_tokens=160000, payload_size=replay_size)
    assert result == before
    assert comp.compression_count == 0
    assert not comp._last_compression_made_progress


def test_estimator_capability_keeps_legacy_compression_plugins_working():
    from agent.conversation_compression import _supported_compression_kwargs

    def legacy(messages, current_tokens=None):
        return messages
    kwargs = _supported_compression_kwargs(legacy, current_tokens=123,
              focus_topic=None, force=False, memory_context='', token_estimator=len)
    assert kwargs == {'current_tokens': 123}


def test_codex_reference_limits_and_explicit_long_context():
    # Codex 0.153.4 openai_models.rs::model_context_window_limits_preserve_their_distinct_meanings.
    standard = ContextCompressor(model='gpt-5.6-sol', config_context_length=272000,
                                 max_tokens=8192, quiet_mode=True)
    assert standard.threshold_tokens == 244800
    assert not standard.should_compress(244799)
    assert standard.should_compress(244800)
    assert standard.threshold_tokens < 258400  # 95% usable window is NOT the trigger.
    expanded = ContextCompressor(model='gpt-5.6-sol', config_context_length=1000000,
                                 threshold_tokens_cap=900000, quiet_mode=True)
    assert expanded.threshold_tokens == 900000
    expanded.update_model('smaller-route', context_length=128000)
    assert expanded.threshold_tokens == 115200


def test_randomized_codex_budget_never_exceeds_window_or_explicit_cap():
    import random
    rng = random.Random(480)
    c = ContextCompressor(model='test', config_context_length=272000, quiet_mode=True)
    for _ in range(1000):
        window = rng.randint(8192, 1050000)
        cap = rng.choice([None, 244800, 900000])
        reserve = rng.randint(1, window - 1)
        c.threshold_tokens_cap = cap
        c.update_model('test', context_length=window, max_tokens=reserve)
        assert c.threshold_tokens == max(1, min(int(window * .9), window - reserve,
                                                cap if cap else window))


def test_provider_pressure_state_transitions():
    import random
    comp = ContextCompressor(model='test/model', config_context_length=272000, quiet_mode=True)
    rng = random.Random(480)
    expected = 0
    for _ in range(1000):
        op = rng.choice(['usage', 'missing', 'compact', 'rough'])
        if op == 'usage':
            expected = rng.randrange(1, 300000)
            comp.update_from_response({'prompt_tokens': expected, 'completion_tokens': 90000,
                                      'reasoning_tokens': 89900})
        elif op == 'missing':
            comp.update_from_response({})
            expected = 0
        elif op == 'compact':
            comp.awaiting_real_usage_after_compression = True
            expected = 0
        assert comp.automatic_compaction_tokens == expected
        assert comp.should_defer_preflight_to_real_usage(rng.randrange(1, 3000000)) == (expected < comp.threshold_tokens)
