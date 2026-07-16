import json
import textwrap

import pytest

from agent import secret_scope
from tools import terminal_tool as terminal_tool_module


@pytest.mark.parametrize(
    "runtime_path",
    [
        '"$ZETTLAB_PRESETS_DIR/skills/gmail/scripts/connector_runtime.py"',
        "${ZETTLAB_PRESETS_DIR}/skills/gmail/scripts/connector_runtime.py",
    ],
    ids=["quoted_plain", "unquoted_braced"],
)
def test_terminal_flow_keeps_direct_runner_after_shared_ancestor_changes(
    monkeypatch,
    tmp_path,
    runtime_path,
):
    """Exercise terminal_tool -> direct subprocess with the real trust checks."""
    script = (
        tmp_path
        / "shared"
        / "presets-v1"
        / "skills"
        / "gmail"
        / "scripts"
        / "connector_runtime.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
        import os

        if os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN") == "flow-secret":
            print("connector-call-succeeded")
        """
    ).lstrip())

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "shared" / "presets-v1"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "flow-secret")
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 424242, raising=False)
    monkeypatch.setattr(terminal_tool_module.os, "getegid", lambda: 424242, raising=False)
    monkeypatch.setattr(terminal_tool_module.os, "getgroups", lambda: [])
    original_writable_check = terminal_tool_module._path_writable_by_current_user

    def trust_test_outer_ancestors(path, *, enforce_cutoff=True):
        # pytest's root is /private/tmp (01777), unlike the production
        # /volume1/subvol chain. Keep the real strict checks for the pinned
        # version tree while isolating this flow to the ancestor timestamp
        # boundary it is intended to exercise.
        if not enforce_cutoff:
            return False
        return original_writable_check(path, enforce_cutoff=True)

    monkeypatch.setattr(
        terminal_tool_module,
        "_path_writable_by_current_user",
        trust_test_outer_ancestors,
    )
    secret_scope.set_multiplex_active(False)

    # A sibling mutation changes the shared ancestor after the root is pinned.
    terminal_tool_module._capture_connector_runtime_root()
    (tmp_path / "shared" / ".recycle").mkdir()

    result = json.loads(terminal_tool_module.terminal_tool(
        f"python3 {runtime_path} list-tools",
        task_id="connector-runtime-shared-ancestor-flow",
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "connector-call-succeeded" in result["output"]
    assert "flow-secret" not in result["output"]
