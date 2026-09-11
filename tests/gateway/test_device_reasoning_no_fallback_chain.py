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
import subprocess
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


APPENDIX_H = (
    "总方案附录 H 条目 H33（内核侧待上游）；上游 issue "
    "https://github.com/NousResearch/hermes-agent/issues/107836"
)
_FAIL_HINT = (
    "设备侧 reasoning override 由 gateway/platforms/zet_agent.py 注入 "
    "agent.request_overrides['extra_body']，而 try_activate_fallback() 不按路由把它归位。"
    "在打包配置里引入 fallback 链会让该 override 在故障转移时发往备用 provider。"
    f"先按 {APPENDIX_H} 归位该 override，再引入 fallback 链。"
)


def _zpk_files():
    if not ZPK_DIR.is_dir():
        pytest.skip("no zpk/ tree in this checkout")
    return [p for p in ZPK_DIR.rglob("*") if p.is_file()]


def _chain_keys_in(text: str):
    return [k for k in FALLBACK_KEYS if k in text]


def test_zpk_config_yaml_defines_no_fallback_chain():
    """`zpk/config/*.yaml` is the config surface HR6 names as authoritative.

    hermes-agent does not ship one: its `zpk/` scripts only prepare
    `$HERMES_HOME` and the service environment (no script writes a
    `config.yaml`), and the device's profile config is created at
    onboarding/runtime under that root, seeded from `cli-config.yaml.example`.

    Absence is therefore recorded as a checked fact, never as proof of safety
    (CR #457 3985625592): the guard asserts that the artifact which *does*
    reach the device is covered, so this branch can never make the suite
    vacuous. If someone starts shipping `zpk/config/`, it is scanned from then
    on.
    """
    cfg_dir = ZPK_DIR / "config"
    shipped = (sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml"))) if cfg_dir.is_dir() else []
    if not shipped:
        assert not cfg_dir.is_dir() or not any(cfg_dir.iterdir()), (
            "zpk/config/ exists with content but no YAML matched — widen this guard"
        )
        # Non-vacuity: the real artifact must still be under guard.
        covered = [seed for seed in CONFIG_SEEDS if (REPO_ROOT / seed).is_file()]
        assert covered, (
            "zpk/config/ is absent AND no config seed is present — nothing is "
            "actually being checked. Point the guard at whatever now provides "
            f"the device config. {_FAIL_HINT}"
        )
        return
    offenders = []
    for path in shipped:
        for key in _chain_keys_in(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {key}")
    assert not offenders, f"打包配置不得含 fallback 链：{'; '.join(offenders)}。{_FAIL_HINT}"


def test_zpk_package_ships_no_fallback_chain():
    """Nothing the device package ships or writes may define a chain.

    Broader than the `zpk/config/*.yaml` check above on purpose: install and
    service scripts can write config too, so the whole packaged tree is read.
    """
    offenders = []
    for path in _zpk_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary payloads carry no config keys
        for key in _chain_keys_in(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {key}")
    assert not offenders, (
        f"设备包不得引入 fallback 链：{'; '.join(offenders)}。{_FAIL_HINT}"
    )


# Files an install path copies verbatim to $HERMES_HOME/config.yaml. This is the
# real profile seed -- `create_profile` clones an existing profile, so the seed is
# what a *fresh* device starts from.
CONFIG_SEEDS = ("cli-config.yaml.example",)
# Install/bootstrap paths that consume the seed. Pinned so a rename cannot quietly
# leave the guard pointing at a file nobody uses any more.
SEED_CONSUMERS = (
    "docker/stage2-hook.sh",
    "scripts/install.sh",
    "scripts/install.ps1",
    "hermes_cli/doctor.py",
)
# Config-ish suffixes, including the `.example` seeds that a plain `*.y*ml` glob
# silently skips (CR #457 3985590592).
_CONFIG_SUFFIXES = (".yaml", ".yml", ".yaml.example", ".yml.example")
# Excluded by *path component*, not prefix: a nested `apps/desktop/node_modules`
# must be skipped too (CR #457 3985590587).
_EXCLUDED_PARTS = frozenset({
    "tests", ".github", "locales", "node_modules", ".venv", "datagen-config-examples",
})


def _tracked_config_files():
    """Git-tracked config files only.

    Enumerating the index rather than walking the filesystem keeps the result
    independent of workspace state: an installed `node_modules` (at any depth)
    or other generated tree can no longer turn this guard red.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    files = []
    for rel in out.split("\0"):
        if not rel:
            continue
        parts = rel.split("/")
        if _EXCLUDED_PARTS.intersection(parts):
            continue
        if rel.endswith(_CONFIG_SUFFIXES):
            files.append(rel)
    return files


def test_config_seed_is_still_where_the_installers_look():
    """Pin the seed itself, so a rename cannot hollow out the next test."""
    for seed in CONFIG_SEEDS:
        assert (REPO_ROOT / seed).is_file(), f"config seed {seed} is missing — update CONFIG_SEEDS"
    for consumer in SEED_CONSUMERS:
        path = REPO_ROOT / consumer
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert any(seed in text for seed in CONFIG_SEEDS), (
            f"{consumer} no longer references {CONFIG_SEEDS} — the seed moved, "
            "point CONFIG_SEEDS at the new one"
        )


def test_config_seed_defines_no_fallback_chain():
    """A fresh install must not start life with a fallback chain.

    `cli-config.yaml.example` is copied verbatim to `$HERMES_HOME/config.yaml`
    by docker/stage2-hook.sh, scripts/install.sh, scripts/install.ps1 and
    hermes_cli.doctor.
    """
    offenders = []
    for seed in CONFIG_SEEDS:
        path = REPO_ROOT / seed
        for key in _chain_keys_in(path.read_text(encoding="utf-8")):
            offenders.append(f"{seed}: {key}")
    assert not offenders, (
        f"配置种子不得含 fallback 链：{'; '.join(offenders)}。{_FAIL_HINT}"
    )


def test_profile_template_defines_no_fallback_chain():
    """Safety net over every tracked config file, seeds included."""
    offenders = []
    for rel in _tracked_config_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for key in _chain_keys_in(text):
            offenders.append(f"{rel}: {key}")
    assert not offenders, (
        f"仓内随包分发的配置 / profile 模板不得含 fallback 链：{'; '.join(offenders)}。{_FAIL_HINT}"
    )


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


# ---------------------------------------------------------------------------
# CR #456 3985250181: agent create/reuse does NOT go through
# `_load_fallback_model()`; both paths call `_refresh_fallback_model()`
# (gateway/run.py, agent reuse and agent create), which re-reads config.yaml
# + managed overlay from disk on every turn. So a chain added *after* the
# gateway started is accepted without a restart.
#
# We cannot block that here: the refresh path lives in `gateway/run.py`, which
# `scripts/test-harness/overlay_gate.json` lists as protected (HR8 kernel), and
# the 2026-09-10 ruling is explicit that HR8 is not to be lifted for this. What
# these tests do instead is pin the behaviour so it is machine-checked and
# nobody can read the guard above as "a runtime chain is impossible".
# ---------------------------------------------------------------------------


class _RunnerStub:
    """`_refresh_fallback_model` only touches `self._fallback_model`."""

    _fallback_model = None


def _refreshed_chain():
    from gateway.run import GatewayRunner

    return GatewayRunner._refresh_fallback_model(_RunnerStub()) or []


def test_refresh_path_reports_no_chain_when_config_has_none(gateway_home):
    _write_cfg(gateway_home, {"model": "pro"})
    assert _refreshed_chain() == []


def test_refresh_path_reports_no_chain_when_config_is_absent(gateway_home):
    assert not (gateway_home / "config.yaml").exists()
    assert _refreshed_chain() == []


@pytest.mark.parametrize("key", ["fallback_model", "fallback_providers"])
def test_refresh_path_accepts_a_chain_added_after_start(gateway_home, key):
    """The residual this guard cannot close, stated as an executable fact.

    `hermes fallback add` (or an overlay push) writes config.yaml while the
    gateway is running; the next agent create/reuse picks it up. If this ever
    stops being true, the prose in this module and in 附录 H H33 is stale and
    must be revisited.
    """
    _write_cfg(gateway_home, {"model": "pro"})
    assert _refreshed_chain() == []

    _write_cfg(gateway_home, {
        "model": "pro",
        key: [{"model": "m", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}],
    })
    chain = _refreshed_chain()
    assert chain, "the refresh path must pick up a chain written after start"
    assert chain[0].get("provider") == "openrouter"


def test_refresh_path_sees_a_managed_overlay_chain(gateway_home, monkeypatch):
    _write_cfg(gateway_home, {"model": "pro"})
    assert _refreshed_chain() == []

    import hermes_cli.managed_scope as managed

    real = managed.apply_managed_overlay

    def _overlay(config):
        merged = dict(real(config) or config or {})
        merged["fallback_providers"] = [
            {"model": "m", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}
        ]
        return merged

    monkeypatch.setattr(managed, "apply_managed_overlay", _overlay)
    assert _refreshed_chain(), "an overlay-pushed chain reaches the refresh path too"
