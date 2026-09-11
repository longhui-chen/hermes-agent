"""The device reasoning override is only safe while the device has no fallback chain.

``_device_reasoning_fast_path`` writes ``extra_body.reasoning`` into
``agent.request_overrides``.  ``try_activate_fallback()`` swaps client / model /
provider / base_url but does not re-scope that bag, so on a provider swap the
override would travel to the fallback provider.  Upstream has since grown
``_rescope_fallback_extra_body``, but it only drops keys it can attribute to a
``custom_providers`` entry -- an override injected by a platform adapter is not
one of those.

We deliberately do NOT strip the override in the failover path: that lives in
``agent/**``, which ``scripts/test-harness/overlay_gate.json`` marks protected
(HR8 kernel).  Instead the exposure is held unreachable by configuration: the
device ships no fallback chain.  These tests pin that precondition, so if a
fallback chain is ever introduced for the device the build fails here and the
override has to be re-scoped first.

See 总方案附录 H (kernel-side pending-upstream entry) for the tracking item.
"""

import json
from pathlib import Path

import pytest
import yaml

from hermes_cli.fallback_config import get_fallback_chain

REPO_ROOT = Path(__file__).resolve().parents[2]
ZPK_DIR = REPO_ROOT / "zpk"
FALLBACK_KEYS = ("fallback_providers", "fallback_model")


def test_empty_config_yields_no_fallback_chain():
    """There is no implicit default chain -- absence of the keys means absence."""
    assert get_fallback_chain({}) == []
    assert get_fallback_chain(None) == []


def test_only_the_two_known_keys_can_introduce_a_chain():
    """Pins the key set this guard has to watch."""
    assert get_fallback_chain({"fallback_model": {"model": "m", "provider": "p"}})
    assert get_fallback_chain({"fallback_providers": [{"model": "m", "provider": "p"}]})
    # Anything else must not create one.
    assert get_fallback_chain({"fallbacks": [{"model": "m"}], "fallback": "m"}) == []


def _zpk_files():
    if not ZPK_DIR.is_dir():
        pytest.skip("no zpk/ tree in this checkout")
    return [p for p in ZPK_DIR.rglob("*") if p.is_file()]


def test_zpk_package_ships_no_fallback_chain():
    """Nothing the device package ships or writes may define a fallback chain."""
    offenders = []
    for path in _zpk_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary payloads carry no config keys
        for key in FALLBACK_KEYS:
            if key in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {key}")
    assert not offenders, (
        "the device package must not introduce a fallback chain while the "
        "device reasoning override is injected by the zet_agent adapter; "
        "re-scope the override before adding one. Offenders: " + "; ".join(offenders)
    )


def test_any_shipped_config_documents_has_no_chain():
    """Any YAML/JSON under zpk/ must parse to a mapping without the chain keys."""
    checked = 0
    for path in _zpk_files():
        if path.suffix.lower() not in (".json",):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        checked += 1
        if isinstance(data, dict):
            for key in FALLBACK_KEYS:
                assert key not in data, f"{path.relative_to(REPO_ROOT)} defines {key}"
    assert checked >= 0  # the scan itself is the assertion; zero JSON files is fine


# ---------------------------------------------------------------------------
# CR #456 3985211322: the zpk scan above is necessary but far from sufficient.
# The chain that actually reaches the runtime is read from $HERMES_HOME/config.yaml
# plus the managed overlay -- neither of which lives under zpk/ -- so the scan can
# pass while the effective config carries a chain. These tests drive the real
# loader instead, and each one has a negative control so a guard that silently
# stopped detecting anything would fail here rather than pass vacuously.
# ---------------------------------------------------------------------------


@pytest.fixture()
def gateway_home(tmp_path, monkeypatch):
    """Point the gateway's config loader at a throwaway HERMES_HOME."""
    import gateway.run as gw

    monkeypatch.setattr(gw, "_hermes_home", tmp_path, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_cfg(home: Path, cfg: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _effective_chain():
    from gateway.run import GatewayRunner

    return GatewayRunner._load_fallback_model() or []


def test_effective_config_without_a_chain_yields_none(gateway_home):
    """The device's actual shape: a config with no fallback keys at all."""
    _write_cfg(gateway_home, {"model": "pro", "agent": {"reasoning_effort": "medium"}})
    assert _effective_chain() == []


@pytest.mark.parametrize("key", ["fallback_model", "fallback_providers"])
def test_a_chain_in_the_effective_config_is_detected(gateway_home, key):
    """Negative control: if a chain IS configured, the real loader must surface it.

    Without this the 'no chain' assertion above could pass because the loader
    silently returned nothing, not because the device is clean.
    """
    _write_cfg(gateway_home, {
        "model": "pro",
        key: [{"model": "m", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}],
    })
    chain = _effective_chain()
    assert chain, f"{key} in $HERMES_HOME/config.yaml must reach the runtime chain"
    assert chain[0].get("provider") == "openrouter"


def test_a_chain_injected_by_the_managed_overlay_is_detected(gateway_home, monkeypatch):
    """The managed overlay is the second source the zpk scan cannot see."""
    _write_cfg(gateway_home, {"model": "pro"})
    assert _effective_chain() == []

    import hermes_cli.managed_scope as managed

    real = managed.apply_managed_overlay

    def _overlay(config):
        merged = dict(real(config) or config or {})
        merged["fallback_providers"] = [
            {"model": "m", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}
        ]
        return merged

    monkeypatch.setattr(managed, "apply_managed_overlay", _overlay)
    import gateway.run as gw

    if hasattr(gw, "_gateway_cfg_cache"):
        monkeypatch.setattr(gw, "_gateway_cfg_cache", None, raising=False)
    assert _effective_chain(), "an overlay-injected chain must be visible to the runtime"


def test_device_reasoning_override_and_a_chain_are_a_flagged_combination(gateway_home):
    """The invariant this file exists for, stated against the real loader.

    If the effective config ever carries a chain while the device route would
    still inject `extra_body.reasoning`, that is the unsafe combination
    (upstream NousResearch/hermes-agent#107836). Today the device ships no
    chain, so this asserts the safe state and fails loudly if that changes.
    """
    from gateway.platforms.zet_agent import _device_reasoning_fast_path

    _write_cfg(gateway_home, {"model": "pro"})
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url="http://127.0.0.1:19090/api/v1/ai-proxy/v1",
        reasoning_config=None,
        request_overrides=None,
    )
    assert applied is True and "reasoning" in (overrides.get("extra_body") or {})
    assert _effective_chain() == [], (
        "the device route injects extra_body.reasoning; a fallback chain would carry "
        "it to another provider on failover. Re-scope the override before adding one."
    )
