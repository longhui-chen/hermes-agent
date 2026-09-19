"""BT tests drive the actual inherited SSE writer, not a simulated emitter."""
import asyncio
import ast
import inspect
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.platforms import api_server
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.platforms.zet_agent_bt import WriterProjection, projection_context


async def write_flow(monkeypatch, items, result, projection=None):
    response = SimpleNamespace(prepare=AsyncMock(), write=AsyncMock())
    monkeypatch.setattr(api_server.web, 'StreamResponse', lambda **_: response)
    adapter = object.__new__(ZetAgentAdapter)
    queue = Queue()
    for item in items:
        queue.put(item)
    queue.put(None)
    task = asyncio.get_running_loop().create_future()
    task.set_result((result, {}))
    token = projection_context.set(projection or WriterProjection())
    try:
        await api_server.APIServerAdapter._write_sse_chat_completion(
            adapter, SimpleNamespace(headers={}), 'completion', 'model', 100, queue, task)
    finally:
        projection_context.reset(token)
    frames = []
    for call in response.write.call_args_list:
        raw = call.args[0].decode()
        for line in raw.splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                frames.append(json.loads(line[6:]))
    return frames


@pytest.mark.asyncio
async def test_writer_suffix_and_delegation_share_ordered_projection_flow(monkeypatch):
    frames = await write_flow(monkeypatch, [
        'hello',
        ('__tool_progress__', {'type':'hermes.delegation.progress','event':'start','subagent_id':'child','status':'running'}),
        ' world',
    ], {'completed':True,'response_transformed':True,'response_transform_suffix':'!','final_response':'hello world!'})
    chunks = [f for f in frames if f.get('choices', [{}])[0].get('delta', {}).get('content')]
    assert ''.join(f['choices'][0]['delta']['content'] for f in chunks) == 'hello world!'
    assert len(chunks) == 3
    assert [f['hermes']['index'] for f in chunks] == [0, 0, 0]
    starts = [f for f in frames if f.get('type') == 'item.started']
    completed = [f for f in frames if f.get('type') == 'item.completed']
    assert len(starts) == len(completed) == 1
    assert frames.index(starts[0]) < frames.index(chunks[0]) < frames.index(completed[0])
    assert completed[-1]['text'] == 'hello world!'
    child = next(f for f in frames if f.get('type') == 'hermes.delegation.progress')
    assert child['index'] == 1 and child['v'] == 1
    assert chunks[0]['hermes']['item_id'] == starts[0]['item_id'] == completed[0]['item_id']


@pytest.mark.asyncio
async def test_writer_early_final_projects_once_flow(monkeypatch):
    frames = await write_flow(monkeypatch, [], {'completed':True, 'final_response':'early answer'})
    text = [f for f in frames if f.get('choices',[{}])[0].get('delta',{}).get('content')]
    assert len(text) == 1
    assert text[0]['hermes']['index'] == 0
    assert any(f.get('type') == 'item.completed' for f in frames)


def test_writer_has_six_original_write_categories():
    tree = ast.parse(inspect.getsource(api_server))
    writer = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_write_sse_chat_completion')
    writes = [n for n in ast.walk(writer) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'write' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'response']
    categories = set()
    for call in writes:
        value = ast.unparse(call.args[0])
        if isinstance(call.args[0], ast.Constant):
            assert call.args[0].value in {b': keepalive\n\n', b'data: [DONE]\n\n'}
            if call.args[0].value.startswith(b'data:'):
                categories.add('done')
        elif 'role_chunk' in value:
            categories.add('role')
        elif 'content_chunk' in value:
            categories.add('content')
        elif 'finish_chunk' in value or 'error_chunk' in value:
            categories.add('terminal')
        elif 'event: hermes.tool.progress' in value:
            categories.add('progress')
        elif 'event: hermes.error' in value:
            categories.add('error')
        else:
            pytest.fail(f'unregistered SSE write: {value}')
    assert categories == {'role', 'content', 'terminal', 'progress', 'error', 'done'}



def test_projection_is_writer_ordered_not_queue_callback_order():
    projection = WriterProjection()
    tool_start = ('__tool_progress__',{'tool':'search','toolCallId':'call','status':'running'})
    tool_result = ('__tool_progress__',{'tool':'search','toolCallId':'call','status':'completed'})
    first = projection.project(tool_start)[0][1]
    text = projection.project('answer')
    last = projection.project(tool_result)[0][1]
    assert first['index'] == last['index'] == 0
    assert next(v for v in text if isinstance(v,str)).wire_fields['hermes']['index'] == 1


def test_rejected_text_uses_single_legacy_fallback_without_recursion():
    projection = WriterProjection()
    value = "x" * (256 * 1024 + 1)
    emitted = projection.project(value)
    assert emitted == [value]


@pytest.mark.asyncio
async def test_writer_classifies_non_frame_queue_values(monkeypatch):
    projection = WriterProjection()
    await write_flow(monkeypatch, [17], {"completed": True}, projection)
    assert projection.sequencer.counters["item_frame_unclassified"] == 1
