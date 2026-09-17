"""AC-1303: real result serializers through the actual BT completion callback.

The JSON-prefix prohibition applies to structured tool results. Plain text is
explicitly allowed a redacted 512-byte preview by D1.
"""
import json
import queue

import pytest
from gateway.platforms.tool_display import result_display
from gateway.platforms.zet_agent_bt import tool_callbacks
from tools.file_operations import ReadResult, PatchResult, SearchResult, SearchMatch
from tools.terminal_tool import _lark_cli_result_json
from tools.todo_tool import TodoStore, todo_tool
from tools.apphost_tool import _ok


def fixtures():
    return [
        ('terminal', {}, _lark_cli_result_json(output='\n'.join(f'line-{n}' for n in range(25)), exit_code=0, timed_out=False), ['Exit code: 0', 'line-24'], ['line-0\n']),
        ('read_file', {'path': '/Users/alice/notes.txt'}, json.dumps(ReadResult(content='\n'.join(f'{n}: text-{n}' for n in range(9)), total_lines=9, file_size=81).to_dict()), ['[PRIVATE_PATH]/notes.txt', 'Lines: 9', '4: text-4'], ['5: text-5']),
        ('patch', {}, json.dumps(PatchResult(success=True, diff='--- a/a.py\n+++ b/a.py\n-old\n+new\n+extra\n', files_modified=['a.py']).to_dict()), ['a.py', '+2 -1'], ['--- a/a.py']),
        ('search_files', {}, json.dumps(SearchResult(matches=[SearchMatch(f'file{n}.md', n, f'title {n}') for n in range(5)], total_count=5).to_dict()), ['Matches: 5', 'file0.md', 'file2.md'], ['file3.md']),
        ('search_files', {}, json.dumps(SearchResult(matches=[SearchMatch('file.md', n, f'title {n}') for n in range(5)], total_count=5).to_dict(densify=True)), ['Matches: 5', 'title 0', 'title 2'], ['title 3']),
        ('nas_search', {}, json.dumps({'total_count': 9, 'results': [{'title': f'Photo {n}'} for n in range(4)]}), ['Matches: 9', 'Photo 2'], ['Photo 3']),
        ('app_host', {'action': 'open'}, _ok({'message': 'Application opened', 'diagnostic': 'not display content'}), ['Action: open', 'Application opened'], ['diagnostic']),
        ('skill_view', {'name': 'demo'}, json.dumps({'success': True, 'name': 'demo', 'content': 'private skill body', 'readiness_status': 'available'}), ['Action: view skill demo', 'available'], ['private skill body']),
        ('todo', {}, todo_tool(store=TodoStore()), ['Action: read tasks', 'total: 0'], ['todos']),
        ('other', {}, json.dumps({'items': [{'title': 'Document found'}], 'token': 'secret-key'}), ['Document found'], ['secret-key']),
    ]


@pytest.mark.parametrize('name,args,raw,expected,forbidden', fixtures())
def test_actual_completion_emits_human_summary(name, args, raw, expected, forbidden):
    q = queue.Queue()
    _, complete = tool_callbacks(q)
    complete('call-one', name, args, raw)
    _, event = q.get_nowait()
    display = event['display']
    summary = display['summary']
    assert not summary.lstrip().startswith(('{', '['))
    assert raw[:64] not in summary
    assert display['content_type'] in {'text', 'markdown', 'error'}
    assert len(summary.encode()) <= 1024
    assert display['bytes'] == len(raw.encode())
    for text in expected:
        assert text in summary
    for text in forbidden:
        assert text not in summary


def test_nested_terminal_json_is_not_a_json_document_preview():
    raw = json.dumps({'exit_code': 0, 'output': json.dumps({'message': 'Completed', 'token': 'secret'})})
    display = result_display(raw, tool_id='terminal')
    assert 'Completed' in display['summary']
    assert '{' not in display['summary'] and 'secret' not in display['summary']


def test_original_multibyte_size_and_bounded_work():
    class BoundedEncode(str):
        def encode(self, *args, **kwargs):
            raise AssertionError('must not encode the whole original result')
    raw = BoundedEncode(json.dumps({'output': '界' * 40000, 'exit_code': 0}, ensure_ascii=False))
    display = result_display(raw, tool_id='terminal')
    assert display['bytes'] == len(str(raw).encode())
    assert display['truncated']
    assert len(display['summary'].encode()) <= 1024


def test_error_and_nonwire_size_conventions():
    raw = json.dumps({'error': 'token=private failure'})
    display = result_display(raw, tool_id='read_file', content_type='json')
    assert display['content_type'] == 'error'
    assert 'private' not in display['summary']
    assert display['bytes'] == len(raw.encode())
    assert 'bytes' not in result_display({'message': 'No original wire encoding'})
    assert 'bytes' not in result_display('\ud800')


def test_plain_text_preview_is_allowed_and_bounded_separately():
    display = result_display('normal text ' * 60)
    assert display['summary'].startswith('Result: normal text')
    assert len(display['summary'].removeprefix('Result: ').encode()) <= 512
    assert display['truncated']
