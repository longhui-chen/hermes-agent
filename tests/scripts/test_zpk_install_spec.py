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

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Extras that must be part of the default ZPK install spec because devices
# cannot lazy-install them (see module docstring).
REQUIRED_ZPK_EXTRAS = {"anthropic", "zpk-runtime"}


def _zpk_install_spec() -> str:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(
        r"^(?:override\s+)?ZPK_INSTALL_SPEC\s*:?=\s*(\S+)",
        makefile,
        re.MULTILINE,
    )
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


def test_zpk_runtime_bootstrap_is_exact_and_hash_locked() -> None:
    optional_dependencies = _optional_dependencies()
    assert optional_dependencies["zpk-runtime"] == [
        "pip==26.1.2",
        "setuptools==83.0.0",
    ]

    with (REPO_ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    locked_packages = {
        package["name"]: package
        for package in lock["package"]
        if package["name"] in {"pip", "setuptools"}
    }
    assert {name: package["version"] for name, package in locked_packages.items()} == {
        "pip": "26.1.2",
        "setuptools": "83.0.0",
    }

    for package in locked_packages.values():
        artifacts = [package["sdist"], *package["wheels"]]
        assert artifacts
        assert all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("hash", ""))
            for artifact in artifacts
        )


def test_zpk_project_build_uses_exact_locked_backend() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["build-system"]["requires"] == ["setuptools==83.0.0"]
    assert (
        "setuptools==83.0.0"
        in pyproject["project"]["optional-dependencies"]["zpk-runtime"]
    )


def test_zpk_payload_checker_rejects_cached_project_wheel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import check_zpk_payload

    source_root = tmp_path / "source"
    source = source_root / "tools" / "runtime.py"
    installed = tmp_path / "site-packages" / "tools" / "runtime.py"
    source.parent.mkdir(parents=True)
    installed.parent.mkdir(parents=True)
    source.write_text("VALUE = 'current'\n", encoding="utf-8")
    installed.write_bytes(source.read_bytes())

    monkeypatch.setattr(check_zpk_payload, "PROJECT_ROOT", source_root)
    monkeypatch.setattr(
        check_zpk_payload,
        "PROJECT_RUNTIME_MODULES",
        {"tools.runtime": Path("tools/runtime.py")},
    )
    monkeypatch.setattr(
        check_zpk_payload.importlib.util,
        "find_spec",
        lambda _module: SimpleNamespace(origin=str(installed)),
    )

    check_zpk_payload._check_project_source_parity()
    installed.write_text("VALUE = 'cached'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="wheel/source parity"):
        check_zpk_payload._check_project_source_parity()


def test_zpk_python_is_explicit_target_compatible_path() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^ZPK_PYTHON\s*\?=\s*(\S+)", makefile, re.MULTILINE)
    assert match, "ZPK_PYTHON must have a reviewed default"
    assert match.group(1) == "/usr/bin/python3.11"
    assert "--python 3.11" not in makefile


@pytest.mark.skipif(os.name == "nt", reason="ZPK Makefile is POSIX-only")
def test_zpk_make_flow_uses_locked_sync_and_reviewed_uv_path(
    tmp_path: Path,
) -> None:
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is unavailable")

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert " pip install" not in makefile
    assert "python -m pip install" not in makefile
    assert makefile.count('"$(UV)" --no-progress') == 6
    (tmp_path / "Makefile").write_text(makefile, encoding="utf-8")

    uv_log = tmp_path / "uv-calls.jsonl"
    fake_uv = tmp_path / "reviewed-uv"
    target_python = tmp_path / "target-python3.11"
    target_python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'Python 3.11.2\'; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    target_python.chmod(0o755)

    fake_uv.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

with Path(os.environ["ZPK_FAKE_UV_LOG"]).open("a", encoding="utf-8") as log:
    log.write(json.dumps({
        "argv": sys.argv[1:],
        "uv_env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("UV_")
        },
    }, sort_keys=True) + "\\n")

if "venv" in sys.argv[1:]:
    venv_index = sys.argv.index("venv")
    venv = Path(sys.argv[venv_index + 1])
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    target = sys.argv[sys.argv.index("--python") + 1]
    python.symlink_to(target)

if (
    "sync" in sys.argv[1:]
    and "--no-install-project" in sys.argv[1:]
    and os.environ.get("ZPK_FAKE_UV_FAIL_DEPENDENCY") == "1"
):
    sys.exit(42)
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    poisoned_uv_env = {
        "UV_BUILD_CONSTRAINT": "/tmp/build-constraint.txt",
        "UV_CONFIG_FILE": "/tmp/uv.toml",
        "UV_CONSTRAINT": "/tmp/constraint.txt",
        "UV_DEFAULT_INDEX": "https://attacker.invalid/simple",
        "UV_EXTRA_INDEX_URL": "https://attacker.invalid/extra",
        "UV_FIND_LINKS": "/tmp/unreviewed-wheels",
        "UV_INDEX": "attacker=https://attacker.invalid/simple",
        "UV_INDEX_STRATEGY": "unsafe-best-match",
        "UV_INDEX_URL": "https://attacker.invalid/simple",
        "UV_INSECURE_HOST": "attacker.invalid",
        "UV_NO_CONFIG": "0",
        "UV_NO_SOURCES": "1",
        "UV_NO_VERIFY_HASHES": "1",
        "UV_OVERRIDE": "/tmp/override.txt",
        "UV_PROJECT": "/tmp/other-project",
        "UV_PROJECT_ENVIRONMENT": "/tmp/other-venv",
        "UV_PYTHON": "/tmp/unreviewed-python",
        "UV_WORKING_DIR": "/tmp/other-project",
    }
    env = os.environ.copy()
    env.update(poisoned_uv_env)
    env["ZPK_FAKE_UV_LOG"] = str(uv_log)

    result = subprocess.run(
        [
            make,
            "--no-print-directory",
            "ZPK_VERBOSE=1",
            f"UV={fake_uv}",
            f"ZPK_PYTHON={target_python}",
            "zpk-venv",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    calls = [
        json.loads(line) for line in uv_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 3
    assert "venv" in calls[0]["argv"]
    python_index = calls[0]["argv"].index("--python")
    assert calls[0]["argv"][python_index + 1] == str(target_python)
    assert "--no-managed-python" in calls[0]["argv"]
    assert "--no-python-downloads" in calls[0]["argv"]

    dependency_sync = calls[1]["argv"]
    project_sync = calls[2]["argv"]
    for sync_argv in (dependency_sync, project_sync):
        assert "sync" in sync_argv
        assert "--locked" in sync_argv
        assert "--no-dev" in sync_argv
        assert "--no-editable" in sync_argv
        assert "pip" not in sync_argv
        assert "install" not in sync_argv
        assert {
            sync_argv[index + 1]
            for index, argument in enumerate(sync_argv[:-1])
            if argument == "--extra"
        } == {"all", "langfuse", "anthropic", "zpk-runtime"}
    assert "--no-install-project" in dependency_sync
    assert "--no-build" in dependency_sync
    assert "--no-build-isolation" not in dependency_sync
    assert "--no-install-project" not in project_sync
    assert "--no-build" not in project_sync
    assert "--no-build-isolation" in project_sync
    reinstall_index = project_sync.index("--reinstall-package")
    assert project_sync[reinstall_index + 1] == "hermes-agent"

    for call in calls:
        assert call["uv_env"]["UV_NO_CONFIG"] == "1"
        for variable in poisoned_uv_env:
            if variable not in {"UV_NO_CONFIG", "UV_PROJECT_ENVIRONMENT"}:
                assert variable not in call["uv_env"]
    assert "UV_PROJECT_ENVIRONMENT" not in calls[0]["uv_env"]
    for call in calls[1:]:
        assert call["uv_env"]["UV_LINK_MODE"] == "copy"
        assert call["uv_env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "venv")

    uv_log.unlink()
    env["ZPK_FAKE_UV_FAIL_DEPENDENCY"] = "1"
    failed_result = subprocess.run(
        [
            make,
            "--no-print-directory",
            "ZPK_VERBOSE=1",
            f"UV={fake_uv}",
            f"ZPK_PYTHON={target_python}",
            "zpk-venv",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed_result.returncode != 0

    failed_calls = [
        json.loads(line) for line in uv_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(failed_calls) == 2
    assert "venv" in failed_calls[0]["argv"]
    assert "--no-install-project" in failed_calls[1]["argv"]
    assert "--no-build" in failed_calls[1]["argv"]
    assert all("--no-build-isolation" not in call["argv"] for call in failed_calls)


def test_uv_lock_check_rejects_project_drift(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    shutil.copyfile(REPO_ROOT / "uv.lock", tmp_path / "uv.lock")

    clean_env = os.environ.copy()
    for variable in (
        "UV_BUILD_CONSTRAINT",
        "UV_CONFIG_FILE",
        "UV_CONSTRAINT",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_INDEX",
        "UV_INDEX_STRATEGY",
        "UV_INDEX_URL",
        "UV_INSECURE_HOST",
        "UV_NO_SOURCES",
        "UV_NO_VERIFY_HASHES",
        "UV_OVERRIDE",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_WORKING_DIR",
    ):
        clean_env.pop(variable, None)
    clean_env["UV_NO_CONFIG"] = "1"

    # Keep the isolated check aligned with the project's lock-time resolver
    # policy. Newer uv releases otherwise treat the relative project setting
    # as removed when invoked from a copied, standalone project.
    lock_policy = [
        "--exclude-newer",
        "14 days",
        "--exclude-newer-package",
        "nemo-relay=false",
        "--exclude-newer-package",
        "vercel=false",
        "--exclude-newer-package",
        "huggingface-hub=false",
    ]

    current = subprocess.run(
        [uv, "lock", "--check", "--offline", *lock_policy],
        cwd=tmp_path,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert current.returncode == 0, current.stdout + current.stderr

    zpk_runtime = 'zpk-runtime = ["pip==26.1.2", "setuptools==83.0.0"]'
    drifted_pyproject = pyproject.replace(
        zpk_runtime,
        f"{zpk_runtime}\ndrift-probe = []",
    )
    assert drifted_pyproject != pyproject
    (tmp_path / "pyproject.toml").write_text(
        drifted_pyproject,
        encoding="utf-8",
    )
    drifted = subprocess.run(
        [uv, "lock", "--check", "--offline", *lock_policy],
        cwd=tmp_path,
        env=clean_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert drifted.returncode != 0
    assert "lock" in (drifted.stdout + drifted.stderr).lower()


# Version group must stop at ';' so PEP 508 environment markers
# ("pkg==1.0; sys_platform != 'win32'") don't leak into the version.
_PIN_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[A-Za-z0-9_,\- ]+\])?\s*==\s*([^\s;]+)"
)


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

    assert extra_pins.get(
        "anthropic"
    ), "anthropic extra no longer pins the anthropic package"
    assert lazy_pins.get(
        "anthropic"
    ), "LAZY_DEPS['provider.anthropic'] no longer pins anthropic"
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
