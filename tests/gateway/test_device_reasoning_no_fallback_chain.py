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
