from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_DEPLOY = REPO_ROOT / "scripts" / "deploy-dev-direct-remote.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_dependency_sync_failure_restores_previous_source(tmp_path: Path) -> None:
    app_root = tmp_path / "apps" / "com.zettlab.claw" / "test-version"
    hermes_src = app_root / "lib" / "hermes-agent"
    venv_bin = hermes_src / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (hermes_src / "old-source-marker").write_text("old", encoding="utf-8")
    (hermes_src / "pyproject.toml").write_text("old-project", encoding="utf-8")
    (hermes_src / "uv.lock").write_text("old-lock", encoding="utf-8")
    _write_executable(venv_bin / "python", "#!/bin/sh\nexit 0\n")

    wrapper = app_root / "bin" / "hermes"
    _write_executable(
        wrapper,
        "#!/bin/sh\nprintf 'bundled plugin\\n'\n",
    )
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir(parents=True)
    hermes_bin.symlink_to(wrapper)

    new_source = tmp_path / "new-source"
    new_source.mkdir()
    (new_source / "new-source-marker").write_text("new", encoding="utf-8")
    (new_source / "pyproject.toml").write_text("new-project", encoding="utf-8")
    (new_source / "uv.lock").write_text("new-lock", encoding="utf-8")
    archive = tmp_path / "hermes-src.new.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        for child in new_source.iterdir():
            bundle.add(child, arcname=child.name)

    calls = tmp_path / "calls.log"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "readlink",
        f"#!/bin/sh\nprintf '%s\\n' '{wrapper}'\n",
    )
    _write_executable(
        fake_bin / "systemctl",
        f"#!/bin/sh\nprintf 'systemctl %s\\n' \"$*\" >> '{calls}'\n"
        "[ \"${1:-}\" = is-active ] && exit 1\n"
        "exit 0\n",
    )
    _write_executable(fake_bin / "journalctl", "#!/bin/sh\nexit 0\n")

    fake_home = tmp_path / "home"
    uv = fake_home / ".local" / "bin" / "uv"
    uv_count = tmp_path / "uv-count"
    _write_executable(
        uv,
        "#!/bin/sh\n"
        f"printf 'uv %s\\n' \"$*\" >> '{calls}'\n"
        f"count=$(cat '{uv_count}' 2>/dev/null || printf 0)\n"
        "count=$((count + 1))\n"
        f"printf '%s' \"$count\" > '{uv_count}'\n"
        '[ "$count" -lt 2 ] && exit 0\n'
        "exit 23\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(fake_home),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HERMES_DEPLOY_ARCHIVE": str(archive),
            "HERMES_DEPLOY_HERMES_BIN": str(hermes_bin),
            "HERMES_DEPLOY_SERVICE": "test-local-server",
            "HERMES_DEPLOY_HEALTH_URL": "http://127.0.0.1:1/health",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DEPLOY)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (hermes_src / "old-source-marker").read_text(encoding="utf-8") == "old"
    assert not (hermes_src / "new-source-marker").exists()
    assert (hermes_src / "venv" / "bin" / "python").is_file()
    assert "deployment failed; restored previous Hermes source" in result.stderr

    command_log = calls.read_text(encoding="utf-8")
    assert (
        "uv sync --frozen --no-dev --no-editable --no-install-project --no-build"
        in command_log
    )
    assert (
        "uv sync --frozen --no-dev --no-editable --no-build-isolation "
        "--reinstall-package hermes-agent"
        in command_log
    )
    assert "--extra zpk-runtime" in command_log
    assert "uv pip install" not in command_log
    assert "systemctl start test-local-server" in command_log
