"""Config `${env:VAR}` SecretRef parity (salvaged from PR #59516).

`${env:VAR}` already resolved in MCP server config (mcp_tool._env_ref_name);
config.yaml's expander treated it as a literal.  These tests pin the parity
plus the cache-snapshot tracking and the non-env-source warning behavior.
"""
from __future__ import annotations

import pytest

from hermes_cli.config import (
    _env_ref_snapshot,
    _env_ref_var_name,
    _expand_env_vars,
)








def test_value_containing_colon_is_not_a_source_ref(monkeypatch):
    """URL-ish or uppercase-colon refs are legacy bare names, not sources —
    only a lowercase ident prefix counts as a SecretRef source."""
    monkeypatch.delenv("MY:WEIRD", raising=False)
    # Uppercase before ':' → treated as a bare (unset) var, kept verbatim,
    # no misleading source warning.
    assert _expand_env_vars("${MY:WEIRD}") == "${MY:WEIRD}"


# ---------------------------------------------------------------------------
# _env_ref_var_name + snapshot tracking
# ---------------------------------------------------------------------------








def test_snapshot_detects_rotation_for_env_prefixed(monkeypatch):
    """The #58514 cache-invalidation contract must hold for ${env:VAR} refs:
    the snapshot records the value under the REAL var name, so a rotation
    changes the snapshot."""
    monkeypatch.setenv("PARITY_ROT", "before")
    snap1 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    monkeypatch.setenv("PARITY_ROT", "after")
    snap2 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    assert snap1 != snap2


def test_expansion_uses_installed_strict_profile_scope(monkeypatch):
    from agent.secret_scope import strict_secret_scope

    monkeypatch.setenv("OPENAI_API_KEY", "dashboard-key")
    with strict_secret_scope({"OPENAI_API_KEY": "worker-key"}):
        assert _expand_env_vars("${env:OPENAI_API_KEY}") == "worker-key"
        assert _expand_env_vars("${OPENAI_API_KEY}") == "worker-key"

    with strict_secret_scope({}):
        assert _expand_env_vars("${env:OPENAI_API_KEY}") == "${env:OPENAI_API_KEY}"
        assert _expand_env_vars("${OPENAI_API_KEY}") == "${OPENAI_API_KEY}"


def test_snapshot_tracks_installed_profile_scope(monkeypatch):
    from agent.secret_scope import strict_secret_scope

    monkeypatch.setenv("OPENAI_API_KEY", "dashboard-key")
    with strict_secret_scope({"OPENAI_API_KEY": "worker-key"}):
        snapshot = _env_ref_snapshot({"k": "${env:OPENAI_API_KEY}"})

    assert snapshot == {"OPENAI_API_KEY": "worker-key"}


def test_load_config_cache_tracks_profile_scope_rotation(
    monkeypatch, _isolate_hermes_home
):
    from agent.secret_scope import strict_secret_scope
    from hermes_cli.config import _LOAD_CONFIG_CACHE, get_config_path, load_config

    get_config_path().write_text(
        "tts:\n  provider: openai\n  api_key: ${env:OPENAI_API_KEY}\n",
        encoding="utf-8",
    )
    _LOAD_CONFIG_CACHE.clear()

    with strict_secret_scope({"OPENAI_API_KEY": "worker-key-1"}):
        assert load_config()["tts"]["api_key"] == "worker-key-1"
    with strict_secret_scope({"OPENAI_API_KEY": "worker-key-2"}):
        assert load_config()["tts"]["api_key"] == "worker-key-2"


def test_managed_config_refs_remain_process_env_only(
    monkeypatch, _isolate_hermes_home
):
    from agent.secret_scope import strict_secret_scope
    from hermes_cli import managed_scope
    from hermes_cli.config import _LOAD_CONFIG_CACHE, get_config_path, load_config

    get_config_path().write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "managed-process-key")
    monkeypatch.setattr(
        managed_scope,
        "load_managed_config",
        lambda: {"tts": {"provider": "openai", "api_key": "${OPENAI_API_KEY}"}},
    )
    _LOAD_CONFIG_CACHE.clear()

    with strict_secret_scope({"OPENAI_API_KEY": "worker-key"}):
        assert load_config()["tts"]["api_key"] == "managed-process-key"
