import json
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


def _prepare_script_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    app_root = tmp_path / "current"
    script = app_root / "prepare-claw-service.sh"
    python_path = app_root / "lib" / "hermes-agent" / "venv" / "bin" / "python"
    hermes_home = tmp_path / "data" / "hermes_home"
    env_path = tmp_path / "data" / "secrets" / "zettlab-claw.env"

    python_path.parent.mkdir(parents=True)
    os.symlink(sys.executable, python_path)
    shutil.copy2(repo_root / "zpk" / "prepare-claw-service.sh", script)
    script.chmod(0o755)
    return app_root, hermes_home, env_path


def _script_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ZETTLAB_PRESETS_DIR", None)
    env.update(overrides)
    return env


def test_prepare_claw_service_normalizes_inline_gateway_config(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)

    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "gateway: {host: 127.0.0.1, multiplex_profiles: false}\n",
        encoding="utf-8",
    )

    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    config = (hermes_home / "config.yaml").read_text(encoding="utf-8")

    assert config.count("gateway:") == 1
    assert "gateway:\n" in config
    assert "  host: 127.0.0.1\n" in config
    assert "  multiplex_profiles: true\n" in config
    assert env_path.exists()
    env_text = env_path.read_text(encoding="utf-8")
    assert "ZET_AGENT_KEY=" in env_text
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text


def test_prepare_claw_service_respects_presets_dir_override(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)

    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text


def test_prepare_claw_service_preserves_existing_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)
    env_path.parent.mkdir(parents=True)
    env_path.write_text(f"ZETTLAB_PRESETS_DIR={presets_dir}\nCUSTOM_SAFE=keep-me\n", encoding="utf-8")

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text
    assert "CUSTOM_SAFE=keep-me\n" in env_text


def test_prepare_claw_service_omits_untrusted_world_writable_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)
    presets_dir.chmod(0o777)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    assert "ZETTLAB_PRESETS_DIR=" not in env_path.read_text(encoding="utf-8")


def test_prepare_claw_service_strips_newlines_from_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR="/custom/presets/current\nEVIL=1\r\n"),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "ZETTLAB_PRESETS_DIR=" not in env_text
    assert "\nEVIL=1" not in env_text


def test_zpk_agent_service_names_are_device_facing():
    repo_root = Path(__file__).resolve().parents[2]

    service = (repo_root / "zpk" / "init.d" / "zettlab-claw.service").read_text(encoding="utf-8")
    start_wrapper = (repo_root / "zpk" / "start-claw-service.sh").read_text(encoding="utf-8")
    package_meta = (repo_root / "zpk" / "package.meta").read_text(encoding="utf-8")
    install = (repo_root / "zpk" / "install.sh").read_text(encoding="utf-8")
    start = (repo_root / "zpk" / "init.d" / "start.sh").read_text(encoding="utf-8")
    stop = (repo_root / "zpk" / "init.d" / "stop.sh").read_text(encoding="utf-8")
    uninstall = (repo_root / "zpk" / "uninstall.sh").read_text(encoding="utf-8")

    meta = json.loads(package_meta)

    assert "EnvironmentFile=-__APP_BASE__/data/secrets/zettlab-claw.env" in service
    assert "ExecStart=__APP_BASE__/current/start-claw-service.sh" in service
    assert '"$APP_ROOT/prepare-claw-service.sh"' in start_wrapper
    assert ". \"$ENV_FILE\"" not in start_wrapper
    assert meta["service_name"] == "zettlab-claw"
    assert "restart" not in meta
    assert "systemctl restart zettlab-claw.service" not in install
    assert "systemctl start zettlab-claw.service" in install
    assert "systemctl start zettlab-claw.service" in start
    assert "systemctl stop zettlab-claw.service" in stop
    assert "zettlab-claw.service" in uninstall


def test_zpk_install_removes_legacy_shared_gateway_unit():
    repo_root = Path(__file__).resolve().parents[2]

    install = (repo_root / "zpk" / "install.sh").read_text(encoding="utf-8")
    uninstall = (repo_root / "zpk" / "uninstall.sh").read_text(encoding="utf-8")

    assert "cleanup_legacy_systemd_services" in install
    assert "hermes-agent-mux.service" in install
    assert "systemctl stop \"$legacy_service\"" in install
    assert "systemctl disable \"$legacy_service\"" in install
    assert "LEGACY_GATEWAY_SERVICE_REMOVED=true" in install
    assert "start_replacement_service_after_legacy_cleanup" in install
    assert "hermes-agent-mux.service" in uninstall
