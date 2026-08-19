"""Tests for the ``transform_tool_result`` plugin hook wired into
``model_tools.handle_function_call``.

Mirrors the ``transform_terminal_output`` hook tests from Phase 1 but
targets the generic tool-result seam that runs for every tool dispatch.
"""

import os
from pathlib import Path

import hermes_cli.plugins as plugins_mod
import model_tools
from agent.trusted_tool_result import TrustedToolResult


_UNSET = object()


def _run_handle_function_call(
    monkeypatch,
    *,
    tool_name="dummy_tool",
    tool_args=None,
    dispatch_result='{"output": "original"}',
    invoke_hook=_UNSET,
    turn_id="",
    api_request_id="",
):
    """Drive ``handle_function_call`` with a mocked registry dispatch."""
    from tools.registry import registry

    monkeypatch.setattr(
        registry, "dispatch",
        lambda name, args, **kw: dispatch_result,
    )
    # Skip unrelated side effects (read-loop tracker).
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    if invoke_hook is not _UNSET:
        # Patch the symbol actually imported inside handle_function_call.
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
        # Supplying a custom invoke_hook means the test expects hooks to
        # fire — make has_hook agree so the has_hook gate doesn't skip the
        # post_tool_call / transform_tool_result emit paths.
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)

    return model_tools.handle_function_call(
        tool_name,
        tool_args or {},
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        turn_id=turn_id,
        api_request_id=api_request_id,
        skip_pre_tool_call_hook=True,
    )








def test_first_valid_string_return_replaces_result(monkeypatch):
    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=lambda hook_name, **kw: [None, {"x": 1}, "first", "second"],
    )
    assert out == "first"


def test_hook_receives_expected_kwargs(monkeypatch):
    captured = {}

    def _hook(hook_name, **kwargs):
        if hook_name == "transform_tool_result":
            captured.update(kwargs)
        return []

    out = _run_handle_function_call(
        monkeypatch,
        tool_name="my_tool",
        tool_args={"a": 1, "b": "x"},
        dispatch_result='{"ok": true}',
        invoke_hook=_hook,
    )
    assert out == '{"ok": true}'
    assert captured["tool_name"] == "my_tool"
    assert captured["args"] == {"a": 1, "b": "x"}
    assert captured["result"] == '{"ok": true}'
    assert captured["task_id"] == "t1"
    assert captured["session_id"] == "s1"
    assert captured["tool_call_id"] == "tc1"




def test_post_tool_call_remains_observational(monkeypatch):
    """post_tool_call return values must NOT replace the result."""
    def _hook(hook_name, **kw):
        if hook_name == "post_tool_call":
            # Observers returning a string must be ignored.
            return ["observer return should be ignored"]
        return []

    out = _run_handle_function_call(
        monkeypatch,
        invoke_hook=_hook,
    )
    assert out == '{"output": "original"}'


def test_transform_tool_result_runs_after_post_tool_call(monkeypatch):
    """post_tool_call sees ORIGINAL result; transform_tool_result sees same and may replace."""
    observed = []

    def _hook(hook_name, **kw):
        if hook_name == "post_tool_call":
            observed.append(("post_tool_call", kw["result"]))
            return []
        if hook_name == "transform_tool_result":
            observed.append(("transform_tool_result", kw["result"]))
            return ["rewritten"]
        return []

    out = _run_handle_function_call(
        monkeypatch,
        dispatch_result='{"raw": "value"}',
        invoke_hook=_hook,
    )
    assert out == "rewritten"
    # Both hooks saw the ORIGINAL (untransformed) result.
    assert observed == [
        ("post_tool_call", '{"raw": "value"}'),
        ("transform_tool_result", '{"raw": "value"}'),
    ]


def test_transform_preserves_process_local_trusted_metadata(monkeypatch):
    raw = TrustedToolResult(
        '{"raw":"failure"}',
        terminal_failure_reason="workflow_checkpoint_identity_invalid",
    )

    observed = {}

    def _hook(hook_name, **kwargs):
        if hook_name == "transform_tool_result":
            observed.update(kwargs)
            return ['{"model":"rewritten"}']
        return []

    out = _run_handle_function_call(
        monkeypatch,
        tool_name="terminal",
        dispatch_result=raw,
        invoke_hook=_hook,
        turn_id="turn-42",
        api_request_id="req-42",
    )

    assert isinstance(out, TrustedToolResult)
    assert out == '{"model":"rewritten"}'
    assert out.terminal_failure_reason == "workflow_checkpoint_identity_invalid"
    assert observed["session_id"] == "s1"
    assert observed["turn_id"] == "turn-42"
    assert observed["api_request_id"] == "req-42"
    assert observed["result"] == '{"raw":"failure"}'


def test_dispatch_wrapper_is_inside_plugin_boundary(monkeypatch):
    """A privileged dispatch wrapper must not cover plugin execution code."""
    events = []
    privileged = False

    def _dispatch(_tool_name, _args, **_kwargs):
        assert privileged is True
        events.append("registry")
        return '{"raw": "failure"}'

    def _execution_middleware(*, args, next_call, **_context):
        assert privileged is False
        events.append("middleware-before")
        result = next_call(args)
        assert privileged is False
        events.append("middleware-after")
        return result

    def _dispatch_wrapper(_tool_name, _args, dispatch):
        nonlocal privileged
        assert privileged is False
        events.append("wrapper-enter")
        privileged = True
        try:
            result = dispatch()
        finally:
            privileged = False
        events.append(("raw-result", result))
        return result

    def _hook(hook_name, **kwargs):
        assert privileged is False
        if hook_name == "post_tool_call":
            events.append(("post", kwargs["result"]))
        if hook_name == "transform_tool_result":
            events.append(("transform", kwargs["result"]))
            return ['{"model": "rewritten"}']
        return []

    monkeypatch.setattr(model_tools.registry, "dispatch", _dispatch)
    monkeypatch.setattr(
        "hermes_cli.middleware._get_middleware_callbacks",
        lambda kind: [_execution_middleware] if kind == "tool_execution" else [],
    )
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _hook)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    result = model_tools.handle_function_call(
        "dummy_tool",
        {"value": 1},
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
        dispatch_wrapper=_dispatch_wrapper,
    )

    assert result == '{"model": "rewritten"}'
    assert events == [
        "middleware-before",
        "wrapper-enter",
        "registry",
        ("raw-result", '{"raw": "failure"}'),
        "middleware-after",
        ("post", '{"raw": "failure"}'),
        ("transform", '{"raw": "failure"}'),
    ]


def test_transform_tool_result_integration_with_real_plugin(monkeypatch, tmp_path):
    """End-to-end: load a real plugin from HERMES_HOME and verify it rewrites results."""
    import yaml

    hermes_home = Path(os.environ["HERMES_HOME"])
    plugins_dir = hermes_home / "plugins"
    plugin_dir = plugins_dir / "transform_result_canon"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: transform_result_canon\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        '    ctx.register_hook("transform_tool_result", '
        'lambda **kw: f\'CANON[{kw["tool_name"]}]\' + kw["result"])\n',
        encoding="utf-8",
    )
    # Plugins are opt-in — must be listed in plugins.enabled to load.
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"plugins": {"enabled": ["transform_result_canon"]}}),
        encoding="utf-8",
    )

    # Force a fresh plugin manager so the new config is picked up.
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins()

    out = _run_handle_function_call(
        monkeypatch,
        tool_name="some_tool",
        dispatch_result='{"payload": 42}',
    )
    assert out == 'CANON[some_tool]{"payload": 42}'
