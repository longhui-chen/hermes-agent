"""Invariants for what the ZPK venv must bundle (ZET-1399).

On ZPK devices the ``tools/lazy_deps.py`` install ladder (uv -> pip ->
ensurepip) is fully broken: there is no system ``uv``, uv-created venvs
ship without ``pip``, and Debian splits ``ensurepip`` into the
``python3.11-venv`` package which the device image does not install.
Anything a device needs at runtime therefore MUST be baked into the ZPK
via ``ZPK_INSTALL_SPEC`` in the Makefile — a lazy-only extra is
effectively unavailable on every shipped device.

The contract:

* The ``anthropic`` extra IS baked in. The native Anthropic provider is a
  first-class product feature (custom API endpoints in the app drive it),
  and 2026-06 fleet evidence (ZET-1399) showed every ZPK built after
  anthropic left ``[all]`` shipping without the SDK and unable to
  lazy-install it.

* Version pins for ANY package declared in both ``pyproject.toml``
  extras and ``tools/lazy_deps.py`` must not drift. Two real incidents:
  the 0.86.0 / 0.87.0 anthropic split would have baked a CVE-affected
  SDK (CVE-2026-34450, CVE-2026-34452) into the firmware while lazy
  installs pulled the fixed one; and upstream's v0.14.0 release commit
  reverted the pyproject side of the aiohttp CVE bump
  (CVE-2026-34513/34518/34519/34520/34525) while LAZY_DEPS kept 3.13.4,
  silently re-baking the vulnerable 3.13.3 into every ZPK.

If a future PR removes the extra from ``ZPK_INSTALL_SPEC`` or lets the
pins drift, these tests fail — re-read the ZET-1399 analysis before
changing the expectations.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Extras that must be part of the default ZPK install spec because devices
# cannot lazy-install them (see module docstring).
REQUIRED_ZPK_EXTRAS = {"anthropic"}


def _zpk_install_spec() -> str:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^ZPK_INSTALL_SPEC\s*\?=\s*(\S+)", makefile, re.MULTILINE)
    assert match, "ZPK_INSTALL_SPEC not found in Makefile"
    return match.group(1)


def _spec_extras(spec: str) -> set[str]:
    match = re.search(r"\[([A-Za-z0-9_,\-. ]+)\]", spec)
    if not match:
        return set()
    return {extra.strip() for extra in match.group(1).split(",") if extra.strip()}


def _optional_dependencies() -> dict[str, list[str]]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return pyproject["project"]["optional-dependencies"]


def _lazy_deps_specs() -> dict[str, tuple[str, ...]]:
    # tests/conftest.py puts the repo root on sys.path; tools.lazy_deps is
    # stdlib-only at import time, so no venv deps are needed here.
    from tools.lazy_deps import LAZY_DEPS

    return dict(LAZY_DEPS)


def test_zpk_install_spec_bakes_in_device_required_extras() -> None:
    extras = _spec_extras(_zpk_install_spec())
    missing = REQUIRED_ZPK_EXTRAS - extras
    assert not missing, (
        f"ZPK_INSTALL_SPEC is missing device-required extras {sorted(missing)}; "
        "devices cannot lazy-install them (ZET-1399), so removing the extra "
        "ships firmware without the dependency entirely."
    )


# Version group must stop at ';' so PEP 508 environment markers
# ("pkg==1.0; sys_platform != 'win32'") don't leak into the version.
_PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[A-Za-z0-9_,\- ]+\])?\s*==\s*([^\s;]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins_from(reqs: list[str] | tuple[str, ...]) -> dict[str, set[str]]:
    """Map normalized package name -> set of pinned versions (extras stripped)."""
    pins: dict[str, set[str]] = {}
    for req in reqs:
        match = _PIN_RE.match(req)
        if match:
            pins.setdefault(_normalize(match.group(1)), set()).add(match.group(2))
    return pins


def test_anthropic_pin_matches_lazy_deps() -> None:
    extra_pins = _pins_from(_optional_dependencies()["anthropic"])
    lazy_pins = _pins_from(_lazy_deps_specs()["provider.anthropic"])

    assert extra_pins.get("anthropic"), "anthropic extra no longer pins the anthropic package"
    assert lazy_pins.get("anthropic"), "LAZY_DEPS['provider.anthropic'] no longer pins anthropic"
    assert extra_pins["anthropic"] == lazy_pins["anthropic"], (
        f"anthropic pin drifted: pyproject extra {sorted(extra_pins['anthropic'])} vs "
        f"lazy_deps {sorted(lazy_pins['anthropic'])}. Keep them in sync so the ZPK bundles "
        "the same (CVE-fixed) version that lazy installs would pull."
    )


def test_no_pin_drift_between_pyproject_extras_and_lazy_deps() -> None:
    """Every package pinned in BOTH places must agree on the version.

    pyproject extras drive what gets baked into the ZPK; LAZY_DEPS drives
    what a runtime lazy install pulls. A drift means the device either runs
    a different version than CI tested, or lazy install silently up/downgrades
    an already-bundled package mid-flight.
    """
    pyproject_pins: dict[str, set[str]] = {}
    pin_sources: dict[str, set[str]] = {}
    for extra, reqs in _optional_dependencies().items():
        for pkg, versions in _pins_from(reqs).items():
            pyproject_pins.setdefault(pkg, set()).update(versions)
            pin_sources.setdefault(pkg, set()).add(extra)

    drifts: list[str] = []
    for feature, specs in _lazy_deps_specs().items():
        for pkg, lazy_versions in _pins_from(specs).items():
            extra_versions = pyproject_pins.get(pkg)
            if extra_versions is None:
                continue  # lazy-only package, nothing to cross-check
            if extra_versions != lazy_versions:
                drifts.append(
                    f"{pkg}: lazy_deps[{feature}] pins {sorted(lazy_versions)} but "
                    f"pyproject extras {sorted(pin_sources[pkg])} pin {sorted(extra_versions)}"
                )

    assert not drifts, (
        "version pins drifted between pyproject.toml extras and "
        "tools/lazy_deps.py (update BOTH sides together):\n  - "
        + "\n  - ".join(sorted(drifts))
    )
