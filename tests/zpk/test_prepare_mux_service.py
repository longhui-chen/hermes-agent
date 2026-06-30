import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _readlink_f_available(tmp_path: Path) -> bool:
    return subprocess.run(
        ["readlink", "-f", str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def test_prepare_mux_service_normalizes_inline_gateway_config(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-mux-service.sh uses GNU readlink -f")

    repo_root = Path(__file__).resolve().parents[2]
    app_root = tmp_path / "current"
    script = app_root / "prepare-mux-service.sh"
    python_path = app_root / "lib" / "hermes-agent" / "venv" / "bin" / "python"
    hermes_home = tmp_path / "data" / "hermes_home"

    python_path.parent.mkdir(parents=True)
    os.symlink(sys.executable, python_path)
    shutil.copy2(repo_root / "zpk" / "prepare-mux-service.sh", script)
    script.chmod(0o755)

    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "gateway: {host: 127.0.0.1, multiplex_profiles: false}\n",
        encoding="utf-8",
    )

    subprocess.run([str(script)], check=True, cwd=str(app_root))

    config = (hermes_home / "config.yaml").read_text(encoding="utf-8")
    assert config.count("gateway:") == 1
    assert "gateway:\n" in config
    assert "  host: 127.0.0.1\n" in config
    assert "  multiplex_profiles: true\n" in config
