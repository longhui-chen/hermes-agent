"""Unit tests for the tool-result image-moderation block detector.

When a text-only main model routes a refused image through the vision_analyze
auxiliary tool, the tool returns {"moderation_blocked": true}. The conversation
loop must detect that in the freshly appended tool results and terminate the
turn as a content-policy block instead of feeding it back to the model (which
would leak a reply for a refused image). See _turn_has_tool_moderation_block.
"""

import json

from agent.conversation_loop import _turn_has_tool_moderation_block


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "tool_call_id": "c1", "name": "vision_analyze", "content": content}


def test_detects_blocked_vision_result_indented_json():
    blocked = json.dumps(
        {"success": False, "moderation_blocked": True, "analysis": "内容不合规"},
        indent=2,
        ensure_ascii=False,
    )
    messages = [{"role": "user", "content": "hi"}, _tool_msg(blocked)]
    assert _turn_has_tool_moderation_block(messages, 1) is True


def test_detects_blocked_vision_result_compact_json():
    blocked = json.dumps({"moderation_blocked": True})
    messages = [_tool_msg(blocked)]
    assert _turn_has_tool_moderation_block(messages, 0) is True


def test_ignores_normal_vision_description():
    ok = json.dumps({"success": True, "analysis": "a cat on a sofa"}, ensure_ascii=False)
    messages = [_tool_msg(ok)]
    assert _turn_has_tool_moderation_block(messages, 0) is False


def test_ignores_results_before_start_index():
    # A block from a PRIOR iteration (before start_idx) must not re-trigger
    # termination on a later, clean iteration.
    blocked = json.dumps({"moderation_blocked": True})
    ok = json.dumps({"success": True, "analysis": "fine"})
    messages = [_tool_msg(blocked), _tool_msg(ok)]
    assert _turn_has_tool_moderation_block(messages, 1) is False


def test_moderation_blocked_false_is_not_a_block():
    payload = json.dumps({"success": True, "moderation_blocked": False})
    messages = [_tool_msg(payload)]
    assert _turn_has_tool_moderation_block(messages, 0) is False


def test_substring_fallback_when_content_wrapped_unparseable():
    # A guardrail observation appended after the JSON breaks json.loads, but the
    # block signal must still be detected via the substring fallback.
    wrapped = json.dumps({"moderation_blocked": True}) + "\n[guardrail: observed]"
    messages = [_tool_msg(wrapped)]
    assert _turn_has_tool_moderation_block(messages, 0) is True


def test_non_tool_messages_ignored():
    blocked = json.dumps({"moderation_blocked": True})
    messages = [{"role": "assistant", "content": blocked}]
    assert _turn_has_tool_moderation_block(messages, 0) is False
