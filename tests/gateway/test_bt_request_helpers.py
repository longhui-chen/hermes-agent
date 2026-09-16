"""BT request parsing must not activate the Zet adapter in a plain API server."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("first", ["api_server", "bt_request_helpers"])
def test_plain_api_request_flow_preserves_cron_functions_and_import_boundary(first, tmp_path):
    script = textwrap.dedent('''
        import asyncio
        import importlib
        import sys
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        import cron.scheduler as scheduler
        original_mark = scheduler.mark_job_run
        original_run = scheduler.run_job

        def assert_boundary():
            assert scheduler.mark_job_run is original_mark
            assert scheduler.run_job is original_run
            assert not any(name.startswith("gateway.platforms.zet_agent") for name in sys.modules)

        assert_boundary()
        importlib.import_module("gateway.platforms." + sys.argv[1])
        from gateway.platforms import api_server, bt_request_helpers
        assert_boundary()
        assert bt_request_helpers._extract_turn_id({"metadata": {"turn_id": "turn-1"}}) == "turn-1"
        assert api_server._extract_plan_ack({"metadata": {
            "turn_id": "turn-1", "plan_ack": {"status": "confirmed"}
        }})["turn_id"] == "turn-1"
        assert_boundary()

        async def exercise_plain_api():
            from gateway.config import PlatformConfig
            adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
            app = web.Application()
            app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
            app.router.add_post("/canonical", adapter._handle_canonical_final_chat_completions)
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "metadata": {"turn_id": "turn-1", "execution_policy": "silent_automation"},
                })
                assert response.status == 400, await response.text()
                assert "silent_automation requires stream=false" in await response.text()
                assert_boundary()
                response = await client.post("/canonical", json={"messages": []})
                assert response.status == 400, await response.text()
                assert "canonical-final-v1 requires" in await response.text()
                assert_boundary()

        asyncio.run(exercise_plain_api())
        assert_boundary()
    ''')
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "hermes"))
    result = subprocess.run(
        [sys.executable, "-c", script, first],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_request_helper_module_has_only_stdlib_imports_and_pure_definitions():
    path = Path(__file__).resolve().parents[2] / "gateway/platforms/bt_request_helpers.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(module.split(".")[0] in sys.stdlib_module_names for module in modules)
        elif isinstance(node, ast.FunctionDef):
            assert not node.decorator_list
        else:
            assert isinstance(node, ast.Assign)
            for expression in ast.walk(node.value):
                if isinstance(expression, ast.Call):
                    assert isinstance(expression.func, ast.Name) and expression.func.id == "frozenset"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(module.split(".")[0] in sys.stdlib_module_names for module in modules)


@pytest.mark.parametrize("fields", [{}, {"item_id": None, "index": None}, {"item_id": "item-1"}, {"index": 0}, {"item_id": "item-1", "index": 0}])
def test_tool_completion_preserves_legacy_payload_except_present_item_fields(fields):
    from gateway.platforms.api_server import _tool_completion_payload

    baseline = _tool_completion_payload("call-1", "search", "found")
    assert "item_id" not in baseline and "index" not in baseline
    result = _tool_completion_payload("call-1", "search", "found", **fields)
    assert result == {**baseline, **{key: value for key, value in fields.items() if value is not None}}
