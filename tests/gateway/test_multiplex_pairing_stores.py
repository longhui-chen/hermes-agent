"""Regression: per-profile PairingStore creation in _start_secondary_profile_adapters.

``gateway/run.py`` referenced ``PairingStore`` at method scope in
``_start_secondary_profile_adapters`` while the class's only import was
method-local inside ``__init__`` — a ``NameError`` at runtime, silently
swallowed by the enclosing ``try/except``, so multiplexing gateways never
created per-profile pairing stores and authz pairing checks for secondary
profiles fell through to the global whitelist.

These tests drive the REAL method (bound onto a bare runner) with the
profile-enumeration and adapter-startup collaborators stubbed, and assert
the per-profile stores actually materialize.
"""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.run import GatewayRunner


def _bare_runner(multiplex: bool = True):
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=multiplex)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_stores = {}
    return runner


def test_secondary_profile_pairing_stores_created(tmp_path, monkeypatch):
    """The served-profiles loop must create a PairingStore per profile.

    Pre-fix this silently did nothing: the ``PairingStore(profile=name)``
    reference raised NameError inside the swallowed try/except.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("coder", tmp_path / ".hermes" / "profiles" / "coder"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["coder"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    # Both the active profile and the served secondary get a store.
    assert "default" in runner.pairing_stores, (
        "active profile PairingStore missing — the NameError swallow is back"
    )
    assert "coder" in runner.pairing_stores, (
        "secondary profile PairingStore missing — the NameError swallow is back"
    )
    assert runner.pairing_stores["default"]._dir in {
        tmp_path / ".hermes" / "pairing",
        tmp_path / ".hermes" / "platforms" / "pairing",
    }
    assert not (tmp_path / ".hermes" / "profiles" / "default").exists()


def test_pairing_store_scoped_to_profile_dir(tmp_path, monkeypatch):
    """The created store must live under the profile's pairing directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
        ("ops", tmp_path / ".hermes" / "profiles" / "ops"),
    ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner._profile_adapters["ops"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    store = runner.pairing_stores["ops"]
    assert store.profile == "ops"
    assert "profiles/ops/pairing" in str(store._dir).replace("\\", "/"), (
        f"store not profile-scoped: {store._dir}"
    )


def test_default_pairing_store_uses_root_when_active_profile_is_named(
    tmp_path, monkeypatch
):
    """Multiplex startup must isolate default auth from the active named
    profile even though PAIRING_DIR was bound in the named-profile process."""
    root = tmp_path / ".hermes"
    active_home = root / "profiles" / "coder"
    active_pairing_dir = active_home / "pairing"
    active_pairing_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active_home))

    runner = _bare_runner()

    async def _no_secondary(profile_name, profile_home, claimed):
        return 0

    runner._start_one_profile_adapters = _no_secondary
    runner._adapter_credential_fingerprint = lambda adapter: None

    with patch(
        "gateway.pairing.PAIRING_DIR", active_pairing_dir
    ), patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("default", root)],
    ), patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="coder",
    ):
        runner._profile_adapters["default"] = {}
        asyncio.run(runner._start_secondary_profile_adapters())

    default_store = runner.pairing_stores["default"]
    assert default_store._dir in {
        root / "pairing",
        root / "platforms" / "pairing",
    }
    assert default_store._dir != active_pairing_dir
    assert not (root / "profiles" / "default").exists()
