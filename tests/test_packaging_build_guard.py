"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_artifact(
    kind: str,
    tmp_path,
    *,
    package_build_marker: str | None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the real PEP 517 hook (build_sdist / build_wheel) as a subprocess.

    The wheel and sdist guards live in SEPARATE cmdclass entries in setup.py
    (the bdist_wheel one behind a try/except ImportError), so each hook needs
    its own regression coverage — a passing sdist test proves nothing about
    the wheel path.
    """
    env = os.environ.copy()
    # nix develop exports this too, so it must not grant permission to build
    # a distributable artifact.
    env["NIX_BUILD_TOP"] = "/build/devshell"
    env.pop("HERMES_NIX_BUILD", None)
    env.pop("HERMES_ZPK_BUILD", None)
    if package_build_marker is not None:
        env[package_build_marker] = "1"
    # Redirect setuptools' scratch dirs (build/, *.egg-info) into tmp_path so
    # the allowed-marker build doesn't litter the real worktree.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_{kind}; build_{kind}(r'{out}')".format(
                kind=kind, out=tmp_path
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_artifact_build_rejects_nix_development_shell_environment(kind, tmp_path):
    result = _build_artifact(kind, tmp_path, package_build_marker=None)

    assert result.returncode != 0
    assert "Building wheels or sdists for hermes-agent is not supported" in result.stderr


@pytest.mark.parametrize("package_build_marker", ["HERMES_NIX_BUILD", "HERMES_ZPK_BUILD"])
@pytest.mark.parametrize(
    ("kind", "artifact_glob"),
    [("sdist", "hermes_agent-*.tar.gz"), ("wheel", "hermes_agent-*.whl")],
)
def test_artifact_build_allows_explicit_package_build_marker(
    package_build_marker,
    kind,
    artifact_glob,
    tmp_path,
):
    result = _build_artifact(
        kind,
        tmp_path,
        package_build_marker=package_build_marker,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob(artifact_glob))
