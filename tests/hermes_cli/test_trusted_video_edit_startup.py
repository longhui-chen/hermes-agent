from __future__ import annotations

import sys
from types import SimpleNamespace

from hermes_cli.trusted_video_edit_startup import (
    _is_gateway_run_request,
    prepare_trusted_video_edit_runtime_before_cli_logging,
)


def test_trusted_video_edit_prewarm_is_scoped_to_gateway_run_unit(monkeypatch):
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", "/presets")
    terminal_tool = SimpleNamespace(
        _late_prepare_video_edit_worker_before_terminal=lambda: None
    )
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", terminal_tool)

    assert _is_gateway_run_request(["hermes", "gateway", "run", "--force"])
    assert _is_gateway_run_request(["hermes", "gateway"])
    assert _is_gateway_run_request(["hermes", "gateway", "--accept-hooks"])
    assert not _is_gateway_run_request(["hermes", "gateway", "status"])
    assert not prepare_trusted_video_edit_runtime_before_cli_logging(
        ["hermes", "gateway", "status"]
    )


def test_gateway_cli_prewarms_trusted_worker_before_logging_flow(monkeypatch):
    events: list[str] = []
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", "/presets")
    terminal_tool = SimpleNamespace(
        _late_prepare_video_edit_worker_before_terminal=lambda: events.append(
            "trusted-worker"
        )
    )
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", terminal_tool)

    assert prepare_trusted_video_edit_runtime_before_cli_logging(
        ["python", "-m", "hermes_cli.main", "gateway", "run"]
    )
    events.append("setup-logging")

    assert events == ["trusted-worker", "setup-logging"]


def test_bare_gateway_cli_prewarms_trusted_worker_before_logging(monkeypatch):
    events: list[str] = []
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", "/presets")
    terminal_tool = SimpleNamespace(
        _late_prepare_video_edit_worker_before_terminal=lambda: events.append(
            "trusted-worker"
        )
    )
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", terminal_tool)

    assert prepare_trusted_video_edit_runtime_before_cli_logging(
        ["hermes", "gateway"]
    )
    events.append("setup-logging")

    assert events == ["trusted-worker", "setup-logging"]
