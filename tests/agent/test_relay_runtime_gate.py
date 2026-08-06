"""Embedding gate for the core NeMo Relay runtime."""

from __future__ import annotations

import pytest

from agent import relay_runtime


@pytest.fixture(autouse=True)
def _reset_relay_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    yield
    relay_runtime._reset_for_tests()


@pytest.mark.parametrize("disabled", ["0", "false", "FALSE", "no", "off"])
def test_explicit_disable_uses_noop_without_loading_native_binding(
    monkeypatch,
    disabled: str,
):
    monkeypatch.setenv("HERMES_NEMO_RELAY_CORE_ENABLED", disabled)

    def fail_if_loaded():
        raise AssertionError("disabled Relay must not import nemo_relay")

    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", fail_if_loaded)

    host = relay_runtime.get_host()

    assert isinstance(host, relay_runtime.NoopRelayRuntime)
    assert relay_runtime.get_runtime() is None
    assert "disabled by environment" in host.reason


@pytest.mark.parametrize("enabled", [None, "1", "true", "yes", "on"])
def test_default_and_explicit_enable_keep_upstream_runtime(monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("HERMES_NEMO_RELAY_CORE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("HERMES_NEMO_RELAY_CORE_ENABLED", enabled)
    fake_relay = object()
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake_relay)

    host = relay_runtime.get_host()

    assert isinstance(host, relay_runtime.RelayRuntime)
    assert host.relay is fake_relay
