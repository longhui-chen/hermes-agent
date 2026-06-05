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

* Version pins for lazily installable deps must not drift between
  ``pyproject.toml`` extras and ``tools/lazy_deps.py`` — the 0.86.0 /
  0.87.0 anthropic split would have baked a CVE-affected SDK
  (CVE-2026-34450, CVE-2026-34452) into the firmware while lazy installs
  pulled the fixed one.

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
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from tools.lazy_deps import LAZY_DEPS
    finally:
        sys.path.pop(0)
    return dict(LAZY_DEPS)


def test_zpk_install_spec_bakes_in_device_required_extras() -> None:
    extras = _spec_extras(_zpk_install_spec())
    missing = REQUIRED_ZPK_EXTRAS - extras
    assert not missing, (
        f"ZPK_INSTALL_SPEC is missing device-required extras {sorted(missing)}; "
        "devices cannot lazy-install them (ZET-1399), so removing the extra "
        "ships firmware without the dependency entirely."
    )


def test_anthropic_pin_matches_lazy_deps() -> None:
    extra_reqs = _optional_dependencies()["anthropic"]
    lazy_reqs = _lazy_deps_specs()["provider.anthropic"]

    extra_pins = {req for req in extra_reqs if req.startswith("anthropic==")}
    lazy_pins = {req for req in lazy_reqs if req.startswith("anthropic==")}

    assert extra_pins, "anthropic extra no longer pins the anthropic package"
    assert lazy_pins, "LAZY_DEPS['provider.anthropic'] no longer pins anthropic"
    assert extra_pins == lazy_pins, (
        f"anthropic pin drifted: pyproject extra {sorted(extra_pins)} vs "
        f"lazy_deps {sorted(lazy_pins)}. Keep them in sync so the ZPK bundles "
        "the same (CVE-fixed) version that lazy installs would pull."
    )
